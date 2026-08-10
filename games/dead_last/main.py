"""Launcher for the Dead Last social game scenario."""

from __future__ import annotations

import asyncio
import os
import logging
from collections import Counter
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


class DeadLastEvaluator(SocialGameEndEvaluator):
    """Evaluator that checks Dead Last win conditions."""

    def __call__(
        self, turn_number: int, messages: List[Tuple[str, Message]], **kwargs: Any
    ) -> List[Tuple[str, Tuple[Tuple[str, int | float | bool], str]]]:
        if turn_number >= self.max_turn_number:
            env = kwargs.get("env")
            if env:
                response: List[
                    Tuple[str, Tuple[Tuple[str, int | float | bool], str]]
                ] = [("environment", (("terminated", True), "Max turns reached."))]
                agent_names = list(env.agents)
                for idx, name in enumerate(agent_names):
                    key = f"agent_{idx + 1}"
                    if env.agent_alive.get(name, False):
                        response.append((key, (("complete_rating", 0.0), "Timeout")))
                    else:
                        response.append(
                            (key, (("complete_rating", -1.0), "Eliminated"))
                        )
                return response
            return [("environment", (("terminated", True), "Max turns reached."))]

        env = kwargs.get("env")
        if not env:
            return [("environment", (("terminated", False), ""))]

        # Check if final offers have been resolved
        if env.internal_state.get("game_over", False):
            agent_names = list(env.agents)
            scores = env.internal_state.get("final_scores", {})
            reason = env.internal_state.get("end_reason", "Game over.")
            response = [("environment", (("terminated", True), reason))]
            for idx, name in enumerate(agent_names):
                key = f"agent_{idx + 1}"
                score = scores.get(name, -1.0)
                response.append((key, (("complete_rating", score), "")))
            return response

        return [("environment", (("terminated", False), ""))]


# ============================================================================
# Action Handler
# ============================================================================


class DeadLastActionHandler(ActionHandler):
    """Handles actions for Dead Last."""

    def handle_action(
        self, env: SocialDeductionGame, agent_name: str, action: AgentAction
    ) -> None:
        if not isinstance(env, DeadLastEnv):
            return

        if action.action_type not in ["action", "speak"]:
            return

        if env.current_state == "Point":
            arg = action.argument.strip().lower()
            # Parse "point NAME"
            target = None
            words = action.argument.split()
            for w in words:
                # Match against alive agent names (case-insensitive)
                for name in env.agents:
                    if w.lower() == name.lower() and env.agent_alive.get(name, False):
                        target = name
                        break
                if target:
                    break

            if target and target != agent_name:
                if "points" not in env.internal_state:
                    env.internal_state["points"] = {}
                env.internal_state["points"][agent_name] = target
                env.recv_message(
                    "Environment",
                    SimpleMessage(message=f"[Game] {agent_name} points at {target}."),
                )
            elif target == agent_name:
                env.recv_message(
                    "Environment",
                    SimpleMessage(
                        message=f"[Game] {agent_name}: You cannot point at yourself."
                    ),
                    receivers=[agent_name],
                )

        elif env.current_state == "Final_offer":
            arg = action.argument.strip().lower()
            # Parse "offer X Y" for 2 players, or "offer X Y Z" for 3
            alive_players = [n for n in env.agents if env.agent_alive.get(n, False)]
            parts = arg.split()

            # Try to extract numbers
            numbers = []
            for p in parts:
                try:
                    numbers.append(int(p))
                except ValueError:
                    continue

            if len(numbers) == len(alive_players):
                if "final_offers" not in env.internal_state:
                    env.internal_state["final_offers"] = {}
                # Map numbers to alive players in order
                offer = {}
                for i, player in enumerate(alive_players):
                    offer[player] = numbers[i]
                env.internal_state["final_offers"][agent_name] = offer

    def get_action_instruction(self, env: SocialDeductionGame, agent_name: str) -> str:
        if not isinstance(env, DeadLastEnv):
            return ""

        alive_players = [n for n in env.agents if env.agent_alive.get(n, False)]
        alive_str = ", ".join(alive_players)

        if env.current_state == "Point":
            return (
                f"You must point at someone to eliminate. Alive players: [{alive_str}]. "
                f"Use 'point NAME' (e.g., 'point Bob'). You cannot point at yourself."
            )
        elif env.current_state == "Discussion":
            alive_count = len(alive_players)
            return (
                f"Discussion phase. {alive_count} players alive: [{alive_str}]. "
                f"Discuss who to target. A strict majority (>{alive_count // 2}) "
                f"pointing at the same person will eliminate them."
            )
        elif env.current_state == "Final_negotiation":
            return (
                f"FINAL NEGOTIATION! Only {len(alive_players)} players remain: [{alive_str}]. "
                f"Negotiate how to split 100 points. You must agree on the exact split or everyone gets nothing."
            )
        elif env.current_state == "Final_offer":
            return (
                f"Submit your final offer. List {len(alive_players)} numbers that sum to 100, "
                f"one for each alive player in this order: [{alive_str}]. "
                f"Use 'offer X Y' (e.g., 'offer 60 40' for 2 players). "
                f"All players must submit identical splits or everyone gets nothing."
            )
        return ""


# ============================================================================
# Environment
# ============================================================================


class DeadLastEnv(SocialDeductionGame):
    """Dead Last game with voting, elimination, and final split negotiation."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(action_handler=DeadLastActionHandler(), **kwargs)

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
            "points": {},
            "round": 1,
            "game_over": False,
            "final_scores": {},
            "end_reason": "",
        }
        return obs

    def _check_eliminations(self) -> None:
        """Apply eliminations based on pointing votes."""
        if not self._should_transition_state():
            return

        if self.current_state == "Point":
            points = self.internal_state.get("points", {})
            if points:
                # Count who each person pointed at
                vote_counts: Dict[str, int] = Counter(points.values())
                alive_count = sum(1 for a in self.agent_alive.values() if a)
                majority_threshold = alive_count // 2 + 1

                # Find target with strict majority
                eliminated = None
                for target, count in vote_counts.items():
                    if count >= majority_threshold:
                        eliminated = target
                        break

                if eliminated:
                    self.agent_alive[eliminated] = False
                    self.recv_message(
                        "Environment",
                        SimpleMessage(
                            message=f"[Game] {eliminated} was eliminated! "
                            f"({vote_counts[eliminated]} out of {alive_count} pointed at them)"
                        ),
                    )
                    _gen_logger.info(
                        f"{eliminated} was eliminated in round {self.internal_state.get('round', '?')}"
                    )
                else:
                    self.recv_message(
                        "Environment",
                        SimpleMessage(
                            message=f"[Game] No majority reached. No one is eliminated this round. "
                            f"(Votes: {dict(vote_counts)})"
                        ),
                    )

                # Clear points for next round
                self.internal_state["points"] = {}
                self.internal_state["round"] = self.internal_state.get("round", 1) + 1

                # Check if we should enter final phase
                alive_players = [
                    n for n in self.agents if self.agent_alive.get(n, False)
                ]
                if len(alive_players) <= 3:
                    self.recv_message(
                        "Environment",
                        SimpleMessage(
                            message=f"[Game] Only {len(alive_players)} players remain: "
                            f"{', '.join(alive_players)}. Entering Final Negotiation!"
                        ),
                    )
                    # Override state to final negotiation
                    self.current_state = "Final_negotiation"
                    if hasattr(self, "_state_turn_count"):
                        self._state_turn_count["Final_negotiation"] = 0
                    if hasattr(self, "_round_robin_idx"):
                        self._round_robin_idx = 0

        elif self.current_state == "Final_offer":
            offers = self.internal_state.get("final_offers", {})
            alive_players = [n for n in self.agents if self.agent_alive.get(n, False)]

            if len(offers) >= len(alive_players):
                # Check if all offers match
                offer_values = list(offers.values())
                all_match = all(offer_values[0] == offer for offer in offer_values[1:])

                if all_match and offer_values:
                    agreed_split = offer_values[0]
                    scores = {}
                    for name in self.agents:
                        if self.agent_alive.get(name, False):
                            scores[name] = agreed_split.get(name, 0) / 100.0
                        else:
                            scores[name] = -1.0

                    self.internal_state["game_over"] = True
                    self.internal_state["final_scores"] = scores
                    self.internal_state["end_reason"] = (
                        f"Players agreed on split: {agreed_split}!"
                    )
                    self.recv_message(
                        "Environment",
                        SimpleMessage(message=f"[Game] Split agreed! {agreed_split}"),
                    )
                else:
                    # No agreement: eliminated players keep their -1.0; survivors get 0.0.
                    scores = {}
                    for name in self.agents:
                        scores[name] = 0.0 if self.agent_alive.get(name, True) else -1.0

                    self.internal_state["game_over"] = True
                    self.internal_state["final_scores"] = scores
                    self.internal_state["end_reason"] = (
                        f"No agreement on split! Offers: {offers}. Everyone gets nothing."
                    )
                    self.recv_message(
                        "Environment",
                        SimpleMessage(
                            message=f"[Game] No agreement! Offers were: {offers}. Everyone gets nothing."
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
) -> DeadLastEnv:
    return DeadLastEnv(
        env_profile=env_profile,
        config=config,
        model_name=model_name,
        evaluators=[DeadLastEvaluator(max_turn_number=80)],
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
    scenario = config.get("description", "Dead Last").format(
        agent_names=", ".join(agent_names)
    )
    env_profile = EnvironmentProfile(
        scenario=scenario,
        relationship=RelationshipType.acquaintance,
        agent_goals=agent_goals,
        tag="dead_last",
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

    parser = argparse.ArgumentParser(description="Run Dead Last game.")
    parser.add_argument("--config", type=str, default=str(CONFIG_PATH))
    parser.add_argument("--roster", type=str, default=str(BASE_DIR / "roster.json"))
    args = parser.parse_args()

    config = load_config(args.config)
    roster = load_config(args.roster)
    config["agents"] = roster.get("agents", [])

    agent_model_name = get_model_names(config)
    env, agents = prepare_scenario("gpt-4o", agent_model_name, config)

    print("Dead Last")
    print("=" * 60)
    print_roster(config)
    print("=" * 60)

    await arun_one_episode(
        env=env,
        agent_list=agents,
        omniscient=False,
        script_like=False,
        json_in_script=False,
        tag="test_dead_last",
        push_to_db=True,
    )


if __name__ == "__main__":
    LOG_FILE = BASE_DIR / "dead_last_game_debug.log"
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
