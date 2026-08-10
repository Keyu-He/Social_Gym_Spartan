"""Launcher for the Public Goods Game."""

from __future__ import annotations

import asyncio
import os
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import redis
from redis_om import get_redis_connection
from rich.logging import RichHandler

from sotopia.agents import LLMAgent
from sotopia.agents.llm_agent import Agents
from sotopia.database.persistent_profile import (
    AgentProfile,
    EnvironmentProfile,
    RelationshipType,
)
from sotopia.envs.evaluators import SocialGameEndEvaluator
from sotopia.envs.social_game import (
    ActionHandler,
    SocialDeductionGame,
    SOCIAL_GAME_PROMPT_TEMPLATE,
    load_config,
)
from sotopia.messages import AgentAction, Message, Observation, SimpleMessage
from sotopia.server import arun_one_episode

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"

os.environ.setdefault("REDIS_OM_URL", "redis://:@localhost:6379")
redis.Redis(host="localhost", port=6379)

_gen_logger = logging.getLogger("sotopia.generation")
_env_logger = logging.getLogger("sotopia.envs.social_game")


class PublicGoodsEvaluator(SocialGameEndEvaluator):
    def __call__(
        self, turn_number: int, messages: List[Tuple[str, Message]], **kwargs: Any
    ) -> List[Tuple[str, Tuple[Tuple[str, int | float | bool], str]]]:
        env = kwargs.get("env")
        if env and env.internal_state.get("game_over", False):
            scores = env.internal_state.get("final_scores", {})
            reason = env.internal_state.get("end_reason", "Game over")
            response: List[Tuple[str, Tuple[Tuple[str, int | float | bool], str]]] = [
                ("environment", (("terminated", True), reason))
            ]
            for idx, name in enumerate(env.agents):
                score = scores.get(name, 0.0)
                response.append((f"agent_{idx + 1}", (("complete_rating", score), "")))
            return response
        if turn_number >= self.max_turn_number:
            return [("environment", (("terminated", True), "Timeout"))]
        return [("environment", (("terminated", False), ""))]


class PublicGoodsActionHandler(ActionHandler):
    def handle_action(
        self, env: SocialDeductionGame, agent_name: str, action: AgentAction
    ) -> None:
        if not isinstance(env, PublicGoodsEnv):
            return
        if action.action_type not in ["action", "speak"]:
            return
        if env.current_state == "Contribute":
            arg = action.argument.strip()
            numbers = re.findall(r"\d+", arg)
            if numbers:
                amount = int(numbers[0])
                amount = max(0, min(10, amount))
                env.internal_state["current_contributions"][agent_name] = amount

    def get_action_instruction(self, env: SocialDeductionGame, agent_name: str) -> str:
        if not isinstance(env, PublicGoodsEnv):
            return ""
        if env.current_state == "Contribute":
            round_num = env.internal_state.get("round", 0) + 1
            max_rounds = env._config.get("max_rounds", 10)
            multiplier = env._config.get("multiplier", 1.5)
            scores = env.internal_state.get("scores", {})
            last_contribs = env.internal_state.get("last_contributions", {})
            contrib_info = ""
            if last_contribs:
                contrib_info = (
                    " Last round: "
                    + ", ".join(f"{n}: {c}" for n, c in last_contribs.items())
                    + "."
                )
            num_players = len(env.agents)
            top_n = num_players // 2
            return (
                f"Round {round_num}/{max_rounds}. Earnings: {scores}.{contrib_info} "
                f"Contribute 0-10 tokens. Use 'contribute N'. "
                f"Pool = total x {multiplier}, split equally. You keep uncontributed tokens. "
                f"Top {top_n} earners WIN, bottom {top_n} LOSE. Relative ranking matters."
            )
        return ""


class PublicGoodsEnv(SocialDeductionGame):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(action_handler=PublicGoodsActionHandler(), **kwargs)

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, str] | None = None,
        agents: Agents | None = None,
        omniscient: bool = False,
        lite: bool = False,
        include_background_observations: bool = True,
    ) -> Dict[str, Observation]:
        obs = super().reset(
            seed=seed,
            options=options,
            agents=agents,
            omniscient=omniscient,
            lite=lite,
            include_background_observations=include_background_observations,
        )
        self.internal_state = {
            "round": 0,
            "scores": {name: 0.0 for name in self.agents},
            "current_contributions": {},
            "last_contributions": {},
            "game_over": False,
            "final_scores": {},
            "end_reason": "",
        }
        return obs

    def _check_eliminations(self) -> None:
        if not self._should_transition_state():
            return

        if self.current_state == "Contribute":
            contribs = self.internal_state.setdefault("current_contributions", {})
            # Default-and-continue: missing contributions default to 0 (free-ride)
            # so the round always resolves and stale data can't carry over.
            for agent in self.agents:
                contribs.setdefault(agent, 0)
            scores = self.internal_state["scores"]

            tokens = self._config.get("tokens_per_round", 10)
            multiplier = self._config.get("multiplier", 1.5)
            num_players = len(self.agents)

            total_pool = sum(contribs.values()) * multiplier
            share = total_pool / num_players

            payoffs = {}
            for name in self.agents:
                contrib = contribs.get(name, 0)
                payoff = (tokens - contrib) + share
                payoffs[name] = round(payoff, 1)
                scores[name] = round(scores.get(name, 0) + payoff, 1)

            round_num = self.internal_state["round"] + 1
            self.internal_state["round"] = round_num
            self.internal_state["last_contributions"] = dict(contribs)
            self.internal_state["current_contributions"] = {}

            contrib_str = ", ".join(
                f"{n}: contributed {contribs.get(n, 0)}" for n in self.agents
            )
            payoff_str = ", ".join(f"{n}: earned {payoffs[n]}" for n in self.agents)
            self.recv_message(
                "Environment",
                SimpleMessage(
                    message=f"[Round {round_num}] Contributions: {contrib_str}. "
                    f"Pool: {sum(contribs.values())} x {multiplier} = {total_pool:.1f}. "
                    f"Share: {share:.1f} each. Payoffs: {payoff_str}. "
                    f"Cumulative: {scores}"
                ),
            )

            max_rounds = self._config.get("max_rounds", 10)
            if round_num >= max_rounds:
                # Top n/2 earners win, bottom n/2 lose
                sorted_players = sorted(
                    scores.items(), key=lambda x: x[1], reverse=True
                )
                num_players = len(sorted_players)
                top_n = num_players // 2
                final = {}
                for rank, (name, _) in enumerate(sorted_players):
                    if rank < top_n:
                        final[name] = 1.0
                    else:
                        final[name] = -1.0
                # Handle ties at the boundary
                boundary_score = sorted_players[top_n - 1][1] if top_n > 0 else 0
                if all(s == boundary_score for _, s in sorted_players):
                    final = {n: 0.0 for n in scores}

                reason = (
                    f"Game over after {max_rounds} rounds. Final earnings: {scores}"
                )
                self.internal_state["game_over"] = True
                self.internal_state["final_scores"] = final
                self.internal_state["end_reason"] = reason
                self.recv_message(
                    "Environment",
                    SimpleMessage(message=f"[Game] {reason}"),
                )


# ============================================================================
# Setup helpers
# ============================================================================


def ensure_agent_profile(config: Dict[str, Any]) -> AgentProfile:
    name = config.get("name", "")
    first_name, _, last_name = name.partition(" ")
    if not last_name:
        last_name = ""
    try:
        existing = AgentProfile.find(
            (AgentProfile.first_name == first_name)
            & (AgentProfile.last_name == last_name)
        ).all()
        if existing:
            return AgentProfile.get(existing[0].pk)
    except Exception:
        pass
    profile = AgentProfile(first_name=first_name, last_name=last_name)
    profile.save()
    return profile


def create_environment(
    env_profile: EnvironmentProfile, model_name: str, config: Dict[str, Any]
) -> PublicGoodsEnv:
    return PublicGoodsEnv(
        env_profile=env_profile,
        config=config,
        model_name=model_name,
        evaluators=[PublicGoodsEvaluator(max_turn_number=10)],
        terminal_evaluators=[],
        hide_unknown=True,
    )


def create_agents(
    agent_profiles: List[AgentProfile],
    env_profile: EnvironmentProfile,
    model_name: str | Dict[str, str] | List[str],
    config: Dict[str, Any],
) -> List[LLMAgent]:
    # Load reflection file if specified
    _reflection_file = config.get("reflection_file", "")
    _reflection_text = ""
    _any_needs_reflection = any(
        a.get("include_reflection") for a in config.get("agents", [])
    )
    if _any_needs_reflection and not _reflection_file:
        raise ValueError(
            "Some agents have include_reflection=true but no reflection_file is specified in config"
        )
    if _reflection_file:
        if not os.path.exists(_reflection_file):
            raise FileNotFoundError(f"Reflection file not found: {_reflection_file}")
        with open(_reflection_file) as _rf:
            _reflection_text = _rf.read().strip()

    agents = []
    for idx, profile in enumerate(agent_profiles):
        agent_name = f"{profile.first_name}{' ' + profile.last_name if profile.last_name else ''}"
        role_goal = env_profile.agent_goals[idx]
        role = config.get("agents", [])[idx].get("role", "")
        secrets = config.get("role_secrets", {}).get(role, "")
        filled_template = (
            SOCIAL_GAME_PROMPT_TEMPLATE.replace("{description}", env_profile.scenario)
            .replace("{secret}", f"Your secret info: {secrets}")
            .replace("{goal}", role_goal)
            .replace(
                "{reflection}",
                _reflection_text
                if idx < len(config.get("agents", []))
                and config["agents"][idx].get("include_reflection")
                else "",
            )
        )
        if isinstance(model_name, dict):
            this_agent_model = model_name.get(
                agent_name, model_name.get("default", "gpt-4")
            )
        elif isinstance(model_name, list):
            this_agent_model = model_name[idx]
        else:
            this_agent_model = model_name
        agent = LLMAgent(
            agent_name=agent_name,
            agent_profile=profile,
            model_name=this_agent_model,
            strict_action_constraint=True,
            custom_template=filled_template,
        )
        agent.goal = role_goal
        agents.append(agent)
    return agents


def prepare_scenario(
    env_model_name: str,
    agent_model_name: str | Dict[str, str] | List[str],
    config: Dict[str, Any] | None = None,
) -> tuple[SocialDeductionGame, List[LLMAgent]]:
    if config is None:
        config = load_config(CONFIG_PATH)
    agent_profiles = [ensure_agent_profile(entry) for entry in config.get("agents", [])]
    agent_goals = [
        config.get("role_goals", {}).get(entry.get("role", ""), "")
        for entry in config.get("agents", [])
    ]
    agent_names = [entry.get("name", "") for entry in config.get("agents", [])]
    scenario = config.get("description", "Public Goods Game").format(
        agent_names=", ".join(agent_names)
    )
    env_profile = EnvironmentProfile(
        scenario=scenario,
        relationship=RelationshipType.acquaintance,
        agent_goals=agent_goals,
        tag="public_goods",
    )
    env_profile.save()
    env = create_environment(env_profile, env_model_name, config)
    agents = create_agents(agent_profiles, env_profile, agent_model_name, config)
    return env, agents


def print_roster(config: Dict[str, Any]) -> None:
    print("Participants & roles:")
    for entry in config.get("agents", []):
        print(f" - {entry.get('name')}: {entry.get('role')}")


def get_model_names(config: Dict[str, Any]) -> Dict[str, str]:
    model_map = {}
    for entry in config.get("agents", []):
        name = entry.get("name")
        model = entry.get("agent_model")
        if not name:
            continue
        if not model:
            raise ValueError(f"Agent '{name}' missing 'agent_model' in config.")
        model_map[name] = model
    return model_map


async def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run Public Goods Game.")
    parser.add_argument("--config", type=str, default=str(CONFIG_PATH))
    parser.add_argument("--roster", type=str, default=str(BASE_DIR / "roster.json"))
    args = parser.parse_args()
    config = load_config(args.config)
    roster = load_config(args.roster)
    config["agents"] = roster.get("agents", [])
    agent_model_name = get_model_names(config)
    env, agents = prepare_scenario("gpt-4o", agent_model_name, config)
    print("Public Goods Game")
    print("=" * 60)
    print_roster(config)
    print("=" * 60)
    await arun_one_episode(
        env=env,
        agent_list=agents,
        omniscient=False,
        script_like=False,
        json_in_script=False,
        tag="test_public_goods",
        push_to_db=True,
    )


if __name__ == "__main__":
    LOG_FILE = BASE_DIR / "public_goods_game_debug.log"
    _fh = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
    _fh.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s - %(message)s")
    )
    _gen_logger.setLevel(logging.DEBUG)
    _gen_logger.addHandler(_fh)
    _env_logger.setLevel(logging.INFO)
    _env_logger.addHandler(_fh)
    _env_logger.addHandler(RichHandler())
    asyncio.run(main())
    conn = get_redis_connection()
    conn.connection_pool.disconnect()
    conn.close()
