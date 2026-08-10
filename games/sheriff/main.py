"""Launcher for the Sheriff of Nottingham game."""

from __future__ import annotations

import asyncio
import os
import logging
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


class SheriffEvaluator(SocialGameEndEvaluator):
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


class SheriffActionHandler(ActionHandler):
    def handle_action(
        self, env: SocialDeductionGame, agent_name: str, action: AgentAction
    ) -> None:
        if not isinstance(env, SheriffEnv):
            return
        if action.action_type not in ["action", "speak"]:
            return

        agents = list(env.agents)
        sheriff_idx = env.internal_state.get("sheriff_idx", 0)
        sheriff = agents[sheriff_idx]

        if env.current_state == "Pack":
            if agent_name == sheriff:
                return  # Sheriff doesn't pack
            arg = action.argument.strip().lower()
            if "smuggle" in arg:
                env.internal_state["packs"][agent_name] = "smuggle"
            else:
                env.internal_state["packs"][agent_name] = "honest"

        elif env.current_state == "Inspect":
            if agent_name != sheriff:
                return  # Only Sheriff inspects
            arg = action.argument.strip().lower()
            # Parse decisions: "inspect Alice, pass Bob, inspect Charlie"
            for name in agents:
                if name == sheriff:
                    continue
                name_lower = name.lower()
                if f"inspect {name_lower}" in arg:
                    env.internal_state["inspections"][name] = "inspect"
                elif f"pass {name_lower}" in arg:
                    env.internal_state["inspections"][name] = "pass"
            # Default: pass anyone not mentioned
            for name in agents:
                if name == sheriff:
                    continue
                if name not in env.internal_state["inspections"]:
                    env.internal_state["inspections"][name] = "pass"

    def get_action_instruction(self, env: SocialDeductionGame, agent_name: str) -> str:
        if not isinstance(env, SheriffEnv):
            return ""

        agents = list(env.agents)
        sheriff_idx = env.internal_state.get("sheriff_idx", 0)
        sheriff = agents[sheriff_idx]
        round_num = env.internal_state.get("round", 0)
        total_rounds = env._config.get("total_rounds", 4)
        gold = env.internal_state.get("gold", {})

        if env.current_state == "Pack":
            if agent_name == sheriff:
                return (
                    f"Round {round_num + 1}/{total_rounds}. You are the SHERIFF this round. "
                    f"Gold: {gold}. Wait for others to pack their goods."
                )
            return (
                f"Round {round_num + 1}/{total_rounds}. {sheriff} is the Sheriff. "
                f"Gold: {gold}. "
                f"Choose 'honest' (safe: 3 gold, no risk) or 'smuggle' (risky: 8 gold if not caught, "
                f"but -4 gold penalty if inspected). The Sheriff will decide whether to inspect you."
            )

        elif env.current_state == "Negotiate":
            if agent_name == sheriff:
                non_sheriff = [n for n in agents if n != sheriff]
                return (
                    f"Round {round_num + 1}/{total_rounds}. You are the SHERIFF. "
                    f"Gold: {gold}. "
                    f"Listen to the merchants ({', '.join(non_sheriff)}). "
                    f"They may offer bribes or plead their case. "
                    f"Remember: inspecting an honest merchant costs YOU 2 gold."
                )
            return (
                f"Round {round_num + 1}/{total_rounds}. {sheriff} is the Sheriff. "
                f"Gold: {gold}. "
                f"Negotiate with the Sheriff. You can claim you're honest, "
                f"offer bribes, or try to convince them not to inspect you."
            )

        elif env.current_state == "Inspect":
            if agent_name == sheriff:
                non_sheriff = [n for n in agents if n != sheriff]
                return (
                    f"Round {round_num + 1}/{total_rounds}. You are the SHERIFF. "
                    f"Gold: {gold}. "
                    f"For each merchant, decide: 'inspect NAME' or 'pass NAME'. "
                    f"Include ALL merchants in one response. "
                    f"Merchants: {', '.join(non_sheriff)}. "
                    f"Example: 'inspect {non_sheriff[0]}, pass {non_sheriff[1]}, pass {non_sheriff[2] if len(non_sheriff) > 2 else non_sheriff[0]}'. "
                    f"If you inspect an honest merchant, YOU pay them 2 gold. "
                    f"If you inspect a smuggler, THEY pay you 4 gold."
                )
            return (
                f"Round {round_num + 1}/{total_rounds}. {sheriff} is deciding whether to inspect. "
                f"Gold: {gold}. Waiting for the Sheriff's decision."
            )
        return ""


# ============================================================================
# Environment
# ============================================================================


class SheriffEnv(SocialDeductionGame):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(action_handler=SheriffActionHandler(), **kwargs)

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
            "sheriff_idx": 0,
            "gold": {name: 0 for name in self.agents},
            "packs": {},
            "inspections": {},
            "game_over": False,
            "final_scores": {},
            "end_reason": "",
        }

        agents = list(self.agents)
        sheriff = agents[0]
        self.recv_message(
            "Environment",
            SimpleMessage(
                message=f"[Game] Round 1. {sheriff} is the Sheriff this round."
            ),
        )
        return obs

    def _update_action_mask(self) -> None:
        """Sheriff doesn't pack; non-Sheriff don't inspect."""
        super()._update_action_mask()
        agents = list(self.agents)
        sheriff_idx = self.internal_state.get("sheriff_idx", 0)
        sheriff = agents[sheriff_idx]

        if self.current_state == "Pack":
            self.action_mask[sheriff_idx] = False
        elif self.current_state == "Inspect":
            for idx, name in enumerate(agents):
                self.action_mask[idx] = name == sheriff

    def _check_eliminations(self) -> None:
        if not self._should_transition_state():
            return

        if self.current_state == "Inspect":
            agents = list(self.agents)
            sheriff_idx = self.internal_state.get("sheriff_idx", 0)
            sheriff = agents[sheriff_idx]
            packs = self.internal_state.get("packs", {})
            inspections = self.internal_state.get("inspections", {})
            gold = self.internal_state["gold"]
            honest_val = self._config.get("honest_value", 3)
            smuggle_val = self._config.get("smuggle_value", 8)
            inspect_honest_pen = self._config.get("inspect_honest_penalty", 2)
            inspect_smuggle_pen = self._config.get("inspect_smuggle_penalty", 4)

            results = []
            for name in agents:
                if name == sheriff:
                    continue
                pack = packs.get(name, "honest")
                decision = inspections.get(name, "pass")

                if decision == "inspect":
                    if pack == "honest":
                        # Sheriff pays penalty
                        gold[sheriff] -= inspect_honest_pen
                        gold[name] += honest_val + inspect_honest_pen
                        results.append(
                            f"{name}: packed honest, inspected. "
                            f"{name}+{honest_val + inspect_honest_pen}, "
                            f"{sheriff}-{inspect_honest_pen}"
                        )
                    else:
                        # Smuggler caught
                        gold[name] -= inspect_smuggle_pen
                        gold[sheriff] += inspect_smuggle_pen
                        results.append(
                            f"{name}: CAUGHT smuggling! "
                            f"{name}-{inspect_smuggle_pen}, "
                            f"{sheriff}+{inspect_smuggle_pen}"
                        )
                else:
                    # Passed
                    if pack == "honest":
                        gold[name] += honest_val
                        results.append(
                            f"{name}: packed honest, passed. {name}+{honest_val}"
                        )
                    else:
                        gold[name] += smuggle_val
                        results.append(
                            f"{name}: smuggled successfully! {name}+{smuggle_val}"
                        )

            round_num = self.internal_state["round"] + 1
            self.internal_state["round"] = round_num

            self.recv_message(
                "Environment",
                SimpleMessage(
                    message=f"[Round {round_num} Results] Sheriff: {sheriff}. "
                    + "; ".join(results)
                    + f". Gold: {gold}"
                ),
            )

            # Reset for next round
            self.internal_state["packs"] = {}
            self.internal_state["inspections"] = {}

            total_rounds = self._config.get("total_rounds", 4)
            if round_num >= total_rounds:
                # Game over -- rank by gold. If every player is tied at the same
                # gold total, no one wins/loses (mirrors public_goods behavior).
                sorted_players = sorted(gold.items(), key=lambda x: x[1], reverse=True)
                n_total = len(sorted_players)
                all_tied = n_total > 0 and sorted_players[0][1] == sorted_players[-1][1]
                if all_tied:
                    final = {name: 0.0 for name in gold}
                else:
                    top_names = set(n for n, _ in sorted_players[: n_total // 2])
                    bot_names = set(
                        n for n, _ in sorted_players[n_total - n_total // 2 :]
                    )
                    final = {}
                    for name, g in gold.items():
                        if name in top_names:
                            final[name] = 1.0
                        elif name in bot_names:
                            final[name] = -1.0
                        else:
                            final[name] = 0.0

                reason = (
                    f"Game over after {total_rounds} rounds. "
                    f"Final gold: {gold}. "
                    f"{sorted_players[0][0]} wins with {sorted_players[0][1]} gold!"
                )
                self.internal_state["game_over"] = True
                self.internal_state["final_scores"] = final
                self.internal_state["end_reason"] = reason
                self.recv_message(
                    "Environment", SimpleMessage(message=f"[Game] {reason}")
                )
            else:
                # Rotate Sheriff
                self.internal_state["sheriff_idx"] = round_num % len(agents)
                new_sheriff = agents[self.internal_state["sheriff_idx"]]
                self.recv_message(
                    "Environment",
                    SimpleMessage(
                        message=f"[Game] Round {round_num + 1}. "
                        f"{new_sheriff} is the Sheriff this round."
                    ),
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
) -> SheriffEnv:
    return SheriffEnv(
        env_profile=env_profile,
        config=config,
        model_name=model_name,
        evaluators=[SheriffEvaluator(max_turn_number=50)],
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
    scenario = config.get("description", "Sheriff of Nottingham").format(
        agent_names=", ".join(agent_names)
    )
    env_profile = EnvironmentProfile(
        scenario=scenario,
        relationship=RelationshipType.acquaintance,
        agent_goals=agent_goals,
        tag="sheriff",
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

    parser = argparse.ArgumentParser(description="Run Sheriff of Nottingham game.")
    parser.add_argument("--config", type=str, default=str(CONFIG_PATH))
    parser.add_argument("--roster", type=str, default=str(BASE_DIR / "roster.json"))
    args = parser.parse_args()

    config = load_config(args.config)
    roster = load_config(args.roster)
    config["agents"] = roster.get("agents", [])

    agent_model_name = get_model_names(config)
    env, agents = prepare_scenario("gpt-4o", agent_model_name, config)

    print("Sheriff of Nottingham")
    print("=" * 60)
    print_roster(config)
    print("=" * 60)

    await arun_one_episode(
        env=env,
        agent_list=agents,
        omniscient=False,
        script_like=False,
        json_in_script=False,
        tag="test_sheriff",
        push_to_db=True,
    )


if __name__ == "__main__":
    LOG_FILE = BASE_DIR / "sheriff_game_debug.log"
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
