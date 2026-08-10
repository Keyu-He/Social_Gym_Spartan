"""Launcher for the Bargaining Game (iterated ultimatum)."""

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


# ============================================================================
# Evaluator
# ============================================================================


class BargainingEvaluator(SocialGameEndEvaluator):
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


# ============================================================================
# Action Handler
# ============================================================================


class BargainingActionHandler(ActionHandler):
    def handle_action(
        self, env: SocialDeductionGame, agent_name: str, action: AgentAction
    ) -> None:
        if not isinstance(env, BargainingEnv):
            return
        if action.action_type not in ["action", "speak"]:
            return

        agents = list(env.agents)
        round_num = env.internal_state.get("round", 0)
        proposer_idx = round_num % 2
        proposer = agents[proposer_idx]
        responder = agents[1 - proposer_idx]

        if env.current_state == "Propose":
            if agent_name != proposer:
                return
            # Parse "offer N"
            arg = action.argument.strip().lower()
            match = re.search(r"(\d+)", arg)
            if match:
                keep = int(match.group(1))
                keep = max(0, min(10, keep))
                env.internal_state["current_offer"] = keep

        elif env.current_state == "Respond":
            if agent_name != responder:
                return
            arg = action.argument.strip().lower()
            if "accept" in arg:
                env.internal_state["current_response"] = "accept"
            elif "reject" in arg:
                env.internal_state["current_response"] = "reject"

    def get_action_instruction(self, env: SocialDeductionGame, agent_name: str) -> str:
        if not isinstance(env, BargainingEnv):
            return ""

        agents = list(env.agents)
        round_num = env.internal_state.get("round", 0)
        max_rounds = env._config.get("max_rounds", 10)
        scores = env.internal_state.get("scores", {})
        proposer_idx = round_num % 2
        proposer = agents[proposer_idx]
        responder = agents[1 - proposer_idx]

        if env.current_state == "Propose":
            if agent_name == proposer:
                return (
                    f"Round {round_num + 1}/{max_rounds}. Scores: {scores}. "
                    f"You are the PROPOSER. You have 10 tokens to split. "
                    f"Choose 'offer N' where N is how many you keep (opponent gets 10-N). "
                    f"Example: 'offer 6' means you keep 6, opponent gets 4. "
                    f"Higher total score wins."
                )
            else:
                return (
                    f"Round {round_num + 1}/{max_rounds}. Scores: {scores}. "
                    f"{proposer} is the Proposer this round. Waiting for their offer."
                )
        elif env.current_state == "Respond":
            offer = env.internal_state.get("current_offer")
            if offer is None:
                offer = 5
            if agent_name == responder:
                return (
                    f"Round {round_num + 1}/{max_rounds}. Scores: {scores}. "
                    f"You are the RESPONDER. {proposer} offers to keep {offer} "
                    f"(you would get {10 - offer}). "
                    f"Choose 'accept' or 'reject'. If rejected, both get 0 this round. "
                    f"Higher total score wins."
                )
            else:
                return (
                    f"Round {round_num + 1}/{max_rounds}. Scores: {scores}. "
                    f"Waiting for {responder} to respond to your offer of keeping {offer}."
                )
        return ""


# ============================================================================
# Environment
# ============================================================================


class BargainingEnv(SocialDeductionGame):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(action_handler=BargainingActionHandler(), **kwargs)

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
            "scores": {name: 0 for name in self.agents},
            "current_offer": None,
            "current_response": None,
            "game_over": False,
            "final_scores": {},
            "end_reason": "",
        }
        return obs

    def _update_action_mask(self) -> None:
        """Only the relevant player acts each phase."""
        super()._update_action_mask()
        agents = list(self.agents)
        round_num = self.internal_state.get("round", 0)
        proposer_idx = round_num % 2
        proposer = agents[proposer_idx]
        responder = agents[1 - proposer_idx]

        if self.current_state == "Propose":
            for idx, name in enumerate(agents):
                self.action_mask[idx] = name == proposer
        elif self.current_state == "Respond":
            for idx, name in enumerate(agents):
                self.action_mask[idx] = name == responder

    def _check_eliminations(self) -> None:
        if not self._should_transition_state():
            return

        if self.current_state == "Respond":
            offer = self.internal_state.get("current_offer")
            response = self.internal_state.get("current_response")
            # Default-and-continue: a missing offer is treated as a 50/50 split,
            # a missing response is treated as a rejection. Either way the round
            # always advances instead of stalling until the max-turn timeout.
            if offer is None:
                offer = 5
            if response is None:
                response = "reject"

            agents = list(self.agents)
            round_num = self.internal_state.get("round", 0)
            proposer_idx = round_num % 2
            proposer = agents[proposer_idx]
            responder = agents[1 - proposer_idx]
            scores = self.internal_state["scores"]

            if response == "accept":
                proposer_gain = offer
                responder_gain = 10 - offer
                scores[proposer] += proposer_gain
                scores[responder] += responder_gain
                msg = (
                    f"[Round {round_num + 1}] {proposer} offered to keep {offer}. "
                    f"{responder} ACCEPTED. {proposer}+{proposer_gain}, "
                    f"{responder}+{responder_gain}. Scores: {scores}"
                )
            else:
                msg = (
                    f"[Round {round_num + 1}] {proposer} offered to keep {offer}. "
                    f"{responder} REJECTED. Both get 0. Scores: {scores}"
                )

            self.internal_state["round"] = round_num + 1
            self.internal_state["current_offer"] = None
            self.internal_state["current_response"] = None

            self.recv_message("Environment", SimpleMessage(message=msg))

            max_rounds = self._config.get("max_rounds", 10)
            if self.internal_state["round"] >= max_rounds:
                s1, s2 = scores[agents[0]], scores[agents[1]]
                if s1 > s2:
                    final = {agents[0]: 1.0, agents[1]: -1.0}
                    reason = f"Game over. {agents[0]} wins! Final: {scores}"
                elif s2 > s1:
                    final = {agents[0]: -1.0, agents[1]: 1.0}
                    reason = f"Game over. {agents[1]} wins! Final: {scores}"
                else:
                    final = {agents[0]: 0.0, agents[1]: 0.0}
                    reason = f"Game over. Tie! Final: {scores}"

                self.internal_state["game_over"] = True
                self.internal_state["final_scores"] = final
                self.internal_state["end_reason"] = reason
                self.recv_message(
                    "Environment", SimpleMessage(message=f"[Game] {reason}")
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
) -> BargainingEnv:
    return BargainingEnv(
        env_profile=env_profile,
        config=config,
        model_name=model_name,
        evaluators=[BargainingEvaluator(max_turn_number=25)],
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
    scenario = config.get("description", "Bargaining").format(
        agent_names=", ".join(agent_names)
    )
    env_profile = EnvironmentProfile(
        scenario=scenario,
        relationship=RelationshipType.acquaintance,
        agent_goals=agent_goals,
        tag="bargaining",
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

    parser = argparse.ArgumentParser(description="Run Bargaining game.")
    parser.add_argument("--config", type=str, default=str(CONFIG_PATH))
    parser.add_argument("--roster", type=str, default=str(BASE_DIR / "roster.json"))
    args = parser.parse_args()

    config = load_config(args.config)
    roster = load_config(args.roster)
    config["agents"] = roster.get("agents", [])

    agent_model_name = get_model_names(config)
    env, agents = prepare_scenario("gpt-4o", agent_model_name, config)

    print("Bargaining Game (Iterated Ultimatum)")
    print("=" * 60)
    print_roster(config)
    print("=" * 60)

    await arun_one_episode(
        env=env,
        agent_list=agents,
        omniscient=False,
        script_like=False,
        json_in_script=False,
        tag="test_bargaining",
        push_to_db=True,
    )


if __name__ == "__main__":
    LOG_FILE = BASE_DIR / "bargaining_game_debug.log"
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
