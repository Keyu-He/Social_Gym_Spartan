"""Launcher for the Centipede Game."""

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


class CentipedeEvaluator(SocialGameEndEvaluator):
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


class CentipedeActionHandler(ActionHandler):
    def handle_action(
        self, env: SocialDeductionGame, agent_name: str, action: AgentAction
    ) -> None:
        if not isinstance(env, CentipedeEnv):
            return
        if action.action_type != "action":
            return
        if env.current_state == "Turn":
            agents = list(env.agents)
            current_idx = env.internal_state.get("current_player_idx", 0)
            if agent_name != agents[current_idx]:
                return
            move = action.argument.lower()
            if "take" in move:
                env.internal_state["decision"] = "take"
            elif "pass" in move:
                env.internal_state["decision"] = "pass"

    def get_action_instruction(self, env: SocialDeductionGame, agent_name: str) -> str:
        if not isinstance(env, CentipedeEnv):
            return ""
        if env.current_state == "Turn":
            agents = list(env.agents)
            current_idx = env.internal_state.get("current_player_idx", 0)
            current_player = agents[current_idx]
            node = env.internal_state.get("current_node", 0)
            game_num = env.internal_state.get("game_num", 0) + 1
            num_games = env._config.get("num_games", 4)
            nodes = env._config.get("nodes", [])
            scores = env.internal_state.get("scores", {})

            if agent_name != current_player:
                return (
                    f"Game {game_num}/{num_games}. "
                    f"Waiting for {current_player} to decide. "
                    f"Cumulative scores: {scores}"
                )

            actor = agents[current_idx]
            other = agents[1 - current_idx]
            if node < len(nodes):
                take_payoffs = nodes[node]
                payoff_str = (
                    f"Take now: you ({actor})={take_payoffs[0]}, "
                    f"{other}={take_payoffs[1]}"
                )
            else:
                pass_payoff = env._config.get("pass_through_payoff", [0, 0])
                payoff_str = (
                    f"Final: you ({actor})={pass_payoff[0]}, {other}={pass_payoff[1]}"
                )

            return (
                f"Game {game_num}/{num_games}, Node {node + 1}/{len(nodes)}. "
                f"Your turn! Choose 'take' or 'pass'. "
                f"{payoff_str}. Pass = pot grows but opponent decides next. "
                f"Cumulative scores: {scores}"
            )
        return ""


class CentipedeEnv(SocialDeductionGame):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(action_handler=CentipedeActionHandler(), **kwargs)

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
            "game_num": 0,
            "current_node": 0,
            "current_player_idx": 0,
            "decision": None,
            "scores": {name: 0 for name in self.agents},
            "game_over": False,
            "final_scores": {},
            "end_reason": "",
        }
        agents = list(self.agents)
        num_games = self._config.get("num_games", 4)
        self.recv_message(
            "Environment",
            SimpleMessage(
                message=f"[Game 1/{num_games}] Starting Centipede Game. "
                f"{agents[0]} goes first. Choose 'take' or 'pass'."
            ),
        )
        return obs

    def _update_action_mask(self) -> None:
        super()._update_action_mask()
        if self.current_state == "Turn":
            current_idx = self.internal_state.get("current_player_idx", 0)
            agents = list(self.agents)
            self.action_mask = [False] * len(agents)
            self.action_mask[current_idx] = True

    def _check_eliminations(self) -> None:
        if not self._should_transition_state():
            return

        if self.current_state != "Turn":
            return

        decision = self.internal_state.get("decision")
        if decision is None:
            return

        agents = list(self.agents)
        current_idx = self.internal_state["current_player_idx"]
        node = self.internal_state["current_node"]
        nodes = self._config.get("nodes", [])
        scores = self.internal_state["scores"]
        game_num = self.internal_state["game_num"]

        # Payoffs are acting-player-relative: payoffs[0] is awarded to the
        # player whose turn it is at the node, payoffs[1] to the other player.
        # This makes the alternating first-mover rule actually fair.
        actor = agents[current_idx]
        other = agents[1 - current_idx]

        if decision == "take":
            if node < len(nodes):
                payoffs = nodes[node]
            else:
                payoffs = self._config.get("pass_through_payoff", [0, 0])

            scores[actor] += payoffs[0]
            scores[other] += payoffs[1]

            self.recv_message(
                "Environment",
                SimpleMessage(
                    message=f"[Game {game_num + 1}] {actor} TAKES at node {node + 1}! "
                    f"Payoffs: {actor}={payoffs[0]}, {other}={payoffs[1]}. "
                    f"Cumulative scores: {scores}"
                ),
            )
            self._start_next_game_or_end()

        elif decision == "pass":
            next_node = node + 1
            if next_node >= len(nodes):
                payoffs = self._config.get("pass_through_payoff", [0, 0])
                scores[actor] += payoffs[0]
                scores[other] += payoffs[1]

                self.recv_message(
                    "Environment",
                    SimpleMessage(
                        message=f"[Game {game_num + 1}] {actor} passes at final node! "
                        f"Full cooperation! Payoffs: {actor}={payoffs[0]}, {other}={payoffs[1]}. "
                        f"Cumulative scores: {scores}"
                    ),
                )
                self._start_next_game_or_end()
            else:
                self.internal_state["current_node"] = next_node
                self.internal_state["current_player_idx"] = 1 - current_idx
                self.internal_state["decision"] = None

                next_player = other
                next_payoffs = nodes[next_node] if next_node < len(nodes) else [0, 0]
                self.recv_message(
                    "Environment",
                    SimpleMessage(
                        message=f"[Game {game_num + 1}] {actor} passes! Pot grows. "
                        f"Now {next_player}'s turn at node {next_node + 1}. "
                        f"Take payoffs: {next_player}={next_payoffs[0]}, {actor}={next_payoffs[1]}"
                    ),
                )

                # Reset turn counter so the self-loop gets another turn
                if hasattr(self, "_state_turn_count"):
                    self._state_turn_count[self.current_state] = 0

    def _start_next_game_or_end(self) -> None:
        game_num = self.internal_state["game_num"] + 1
        num_games = self._config.get("num_games", 4)
        agents = list(self.agents)

        if game_num >= num_games:
            scores = self.internal_state["scores"]
            s1, s2 = scores[agents[0]], scores[agents[1]]
            if s1 > s2:
                final = {agents[0]: 1.0, agents[1]: -1.0}
                reason = (
                    f"All {num_games} games done. {agents[0]} wins! Final: {scores}"
                )
            elif s2 > s1:
                final = {agents[0]: -1.0, agents[1]: 1.0}
                reason = (
                    f"All {num_games} games done. {agents[1]} wins! Final: {scores}"
                )
            else:
                final = {agents[0]: 0.0, agents[1]: 0.0}
                reason = f"All {num_games} games done. Tie! Final: {scores}"

            self.internal_state["game_over"] = True
            self.internal_state["final_scores"] = final
            self.internal_state["end_reason"] = reason
            self.recv_message("Environment", SimpleMessage(message=f"[Game] {reason}"))
        else:
            self.internal_state["game_num"] = game_num
            self.internal_state["current_node"] = 0
            self.internal_state["current_player_idx"] = game_num % 2
            self.internal_state["decision"] = None

            first_player = agents[game_num % 2]
            self.recv_message(
                "Environment",
                SimpleMessage(
                    message=f"[Game {game_num + 1}/{num_games}] Starting next Centipede Game. "
                    f"{first_player} goes first."
                ),
            )
            if hasattr(self, "_state_turn_count"):
                self._state_turn_count[self.current_state] = 0


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
) -> CentipedeEnv:
    return CentipedeEnv(
        env_profile=env_profile,
        config=config,
        model_name=model_name,
        evaluators=[CentipedeEvaluator(max_turn_number=30)],
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
    scenario = config.get("description", "Centipede Game").format(
        agent_names=", ".join(agent_names)
    )
    env_profile = EnvironmentProfile(
        scenario=scenario,
        relationship=RelationshipType.acquaintance,
        agent_goals=agent_goals,
        tag="centipede",
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

    parser = argparse.ArgumentParser(description="Run Centipede Game.")
    parser.add_argument("--config", type=str, default=str(CONFIG_PATH))
    parser.add_argument("--roster", type=str, default=str(BASE_DIR / "roster.json"))
    args = parser.parse_args()
    config = load_config(args.config)
    roster = load_config(args.roster)
    config["agents"] = roster.get("agents", [])
    agent_model_name = get_model_names(config)
    env, agents = prepare_scenario("gpt-4o", agent_model_name, config)
    print("Centipede Game")
    print("=" * 60)
    print_roster(config)
    print("=" * 60)
    await arun_one_episode(
        env=env,
        agent_list=agents,
        omniscient=False,
        script_like=False,
        json_in_script=False,
        tag="test_centipede",
        push_to_db=True,
    )


if __name__ == "__main__":
    LOG_FILE = BASE_DIR / "centipede_game_debug.log"
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
