"""Launcher for the Rock, Paper, Scissors game."""

from __future__ import annotations

import asyncio
import logging
import os
import random
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

# Loggers (configured in main if running standalone)
_gen_logger = logging.getLogger("sotopia.generation")
_env_logger = logging.getLogger("sotopia.envs.social_game")


# ============================================================================
# Evaluator
# ============================================================================


class RPSEvaluator(SocialGameEndEvaluator):
    def __call__(
        self, turn_number: int, messages: List[Tuple[str, Message]], **kwargs: Any
    ) -> List[Tuple[str, Tuple[Tuple[str, int | float | bool], str]]]:
        env = kwargs.get("env")
        # The match ends when its rounds are played, not after a fixed number of env steps:
        # a re-asked player takes one extra step in that round.
        max_rounds = env._config.get("max_rounds", 10) if env else 10
        rounds_done = env.internal_state.get("round", 0) if env else 0
        if rounds_done >= max_rounds or turn_number >= self.max_turn_number:
            if env:
                scores = env.internal_state.get("scores", {})
                response: List[
                    Tuple[str, Tuple[Tuple[str, int | float | bool], str]]
                ] = [
                    (
                        "environment",
                        (("terminated", True), f"All {rounds_done} rounds played"),
                    )
                ]

                agent_names = list(env.agents)
                scores_dict = {name: scores.get(name, 0) for name in agent_names}

                # Logic:
                # Highest score wins (1), Loser (-1). Tie -> 0.

                values = list(scores_dict.values())
                rewards = {name: 0.0 for name in agent_names}

                if len(values) == 2:
                    s1, s2 = values[0], values[1]
                    n1, n2 = agent_names[0], agent_names[1]

                    if s1 > s2:
                        rewards[n1] = 1.0
                        rewards[n2] = -1.0
                    elif s2 > s1:
                        rewards[n1] = -1.0
                        rewards[n2] = 1.0

                for agent_name in agent_names:
                    idx = agent_names.index(agent_name)
                    key = f"agent_{idx+1}"
                    raw_score = scores_dict.get(agent_name, 0)
                    reward = rewards.get(agent_name, 0.0)
                    response.append(
                        (
                            key,
                            (("complete_rating", reward), f"Final Score: {raw_score}"),
                        )
                    )
                return response
            return [("environment", (("terminated", True), "Max turns reached"))]
        return [("environment", (("terminated", False), ""))]


# ============================================================================
# Action Handler
# ============================================================================


MOVES = ("Rock", "Paper", "Scissors")
# Singular and plural are the same throw ("rocks", "papers", "scissor").
MOVE_WORDS = {
    "Rock": ("rock", "rocks"),
    "Paper": ("paper", "papers"),
    "Scissors": ("scissors", "scissor"),
}


# A move word right after a negation in its clause ("I will not swerve", "don't defect") does not name that move.
_NEGATION = re.compile(
    r"\b(?:not|no|never|neither|nor|cannot|don'?t|won'?t|can'?t|shouldn'?t|wouldn'?t)\b|n't\b"
)
_CLAUSE_BREAK = re.compile(r"[.;:!?,\n]|\bbut\b")


def is_negated(text: str, start: int) -> bool:
    """True when one of the 3 words before position `start`, in the same clause, is a negation."""
    clause = _CLAUSE_BREAK.split(text[:start])[-1]
    return bool(_NEGATION.search(" ".join(clause.split()[-3:])))


def names(text: str, word: str) -> bool:
    """True when `text` uses `word` as a whole word at least once with no negation right before it."""
    return any(not is_negated(text, m.start()) for m in re.finditer(rf"\b{word}\b", text))


def read_move(argument: str) -> str | None:
    """The throw a reply names, when it names exactly one of the three as a word, else None.

    Whole words only ("paperwork" is not paper), and a negated throw is not named, so "not rock, paper"
    is Paper. A reply naming two throws ("rock beats scissors so I play paper") or none ("not rock") is
    not read as a move; the player is asked again.
    """
    text = argument.lower().replace("’", "'")
    named = [move for move, words in MOVE_WORDS.items() if any(names(text, w) for w in words)]
    return named[0] if len(named) == 1 else None


class RPSActionHandler(ActionHandler):
    def handle_action(
        self, env: SocialDeductionGame, agent_name: str, action: AgentAction
    ) -> None:
        if not isinstance(env, RPSEnv) or agent_name not in env.awaiting():
            return
        move = (
            read_move(action.argument)
            if action.action_type in ["action", "speak"]
            else None
        )
        env.take_reply(agent_name, move)

    def get_action_instruction(self, env: SocialDeductionGame, agent_name: str) -> str:
        text = "You are playing Rock-Paper-Scissors. Choose 'action: rock', 'action: paper', or 'action: scissors'. You cannot speak."
        if env.internal_state.get("unclear", {}).get(agent_name):
            text += (
                " Your last reply did not name exactly one of 'rock', 'paper' or 'scissors', so you are asked again."
                " If this reply is also unclear, your throw is drawn at random."
            )
        return text


# ============================================================================
# Environment
# ============================================================================


class RPSEnv(SocialDeductionGame):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(action_handler=RPSActionHandler(), **kwargs)
        self.internal_state = {"round": 0, "scores": {}, "current_moves": {}}

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
            "scores": {},
            "current_moves": {},
            "unclear": {},
            "drawn": [],
        }
        for agent in self.agents:
            self.internal_state["scores"][agent] = 0
        # Throws drawn for twice-unclear replies; seeded from the roster so replays agree.
        self._fallback_rng = random.Random(f"{self._config.get('seed')}-fallback")
        return obs

    def awaiting(self) -> List[str]:
        """Players whose throw for the current round is not in yet (both, at the start of a round)."""
        moves = self.internal_state.get("current_moves", {})
        return [a for a in self.agents if a not in moves]

    def take_reply(self, agent_name: str, move: str | None) -> None:
        """Hold a clear throw; ask an unclear (or failed) reply once more, then draw the throw at random.

        Throws stay hidden until the round is complete: the state is private, so a player asked
        again never sees the other player's reply.
        """
        if move is not None:
            self.internal_state["current_moves"][agent_name] = move
            return
        unclear = self.internal_state.setdefault("unclear", {})
        unclear[agent_name] = unclear.get(agent_name, 0) + 1
        round_num = self.internal_state.get("round", 0) + 1
        if unclear[agent_name] == 1:
            message = (
                f"[Round {round_num}] {agent_name}'s reply did not name exactly one of 'rock', 'paper' or "
                f"'scissors', so {agent_name} is asked again. Throws stay hidden until the round is complete."
            )
        else:
            self.internal_state["current_moves"][agent_name] = self._fallback_rng.choice(MOVES)
            self.internal_state.setdefault("drawn", []).append(agent_name)
            message = (
                f"[Round {round_num}] {agent_name}'s second reply was also unclear, "
                f"so {agent_name}'s throw is drawn at random."
            )
        self.recv_message("Environment", SimpleMessage(message=message))

    def _update_action_mask(self) -> None:
        super()._update_action_mask()
        if "current_moves" in self.internal_state:
            due = self.awaiting()
            self.action_mask = [a in due for a in self.agents]

    def _should_transition_state(self) -> bool:
        # "Entering state: Round" only when a new round starts: not while a player is asked again, and not after
        # the last round.
        if (
            self.internal_state.get("current_moves")
            or self.internal_state.get("unclear")
            or self.internal_state.get("round", 0) >= self._config.get("max_rounds", 10)
        ):
            return False
        return super()._should_transition_state()

    def _check_eliminations(self) -> None:
        if self.awaiting():
            return  # someone is being asked again; the round resolves once every throw is in
        moves = self.internal_state["current_moves"]
        drawn = self.internal_state.get("drawn", [])
        if len(self.agents) == 2:
            agents = list(self.agents)
            a1, a2 = agents[0], agents[1]
            m1, m2 = moves[a1], moves[a2]

            result, winner = "Draw", None
            if m1 == m2:
                result = "Draw"
            elif (
                (m1 == "Rock" and m2 == "Scissors")
                or (m1 == "Scissors" and m2 == "Paper")
                or (m1 == "Paper" and m2 == "Rock")
            ):
                winner = a1
            else:
                winner = a2

            # The round winner gets a point and the loser none, as the rules text says.
            r1, r2 = (1, 0) if winner == a1 else (0, 1) if winner == a2 else (0, 0)
            result = f"{winner} wins!" if winner else "Draw"

            self.internal_state["scores"][a1] += r1
            self.internal_state["scores"][a2] += r2

            def shown(name: str, move: str) -> str:
                return f"{move}, drawn at random" if name in drawn else move

            msg = f"Round {self.internal_state['round']+1}: {a1} ({shown(a1, m1)}) vs {a2} ({shown(a2, m2)}) -> {result}"
            self.recv_message("Environment", SimpleMessage(message=msg))

            self.internal_state["round"] += 1
            self.internal_state["current_moves"] = {}
            self.internal_state["unclear"] = {}
            self.internal_state["drawn"] = []


# ============================================================================
# Setup helpers
# ============================================================================


def ensure_agent_profile(config: Dict[str, Any]) -> AgentProfile:
    """Create or retrieve agent profile."""
    name = config.get("name", "")
    role = config.get("role", "")
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

    role_secret = config.get("role_secrets", {}).get(role, "")
    profile = AgentProfile(
        first_name=first_name, last_name=last_name, secret=role_secret
    )
    profile.save()
    return profile


def create_environment(
    env_profile: EnvironmentProfile, model_name: str, config: Dict[str, Any]
) -> RPSEnv:
    return RPSEnv(
        env_profile=env_profile,
        config=config,
        model_name=model_name,
        # action_order is handled by config state_properties
        # Safety cap only: every round takes one step, or two when a player is asked again.
        evaluators=[RPSEvaluator(max_turn_number=2 * config.get("max_rounds", 10))],
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
    scenario = config.get("description", "RPS").format(
        agent_names=", ".join(agent_names)
    )

    env_profile = EnvironmentProfile(
        scenario=scenario,
        relationship=RelationshipType.acquaintance,
        agent_goals=agent_goals,
        tag="rps",
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
    """Extract model names from config. Enforces strict requirement."""
    model_map = {}
    for entry in config.get("agents", []):
        name = entry.get("name")
        model = entry.get("agent_model")
        if not name:
            continue
        if not model:
            raise ValueError(
                f"Agent '{name}' missing 'agent_model' in config configuration."
            )
        model_map[name] = model
    return model_map


async def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run Rock-Paper-Scissors game.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(CONFIG_PATH),
        help="Path to configuration file",
    )
    parser.add_argument(
        "--roster",
        type=str,
        default=str(BASE_DIR / "roster.json"),
        help="Path to roster file",
    )
    args = parser.parse_args()

    config_path = args.config
    roster_path = args.roster

    config = load_config(config_path)
    roster = load_config(roster_path)

    # Merge roster into config for consistent access
    config["agents"] = roster.get("agents", [])

    agent_model_name = get_model_names(config)
    env_model_name = "gpt-4o"

    # We pass config explicitly to prepare_scenario
    env, agents = prepare_scenario(env_model_name, agent_model_name, config)

    print("✊✋✌️ Rock Paper Scissors")
    print("=" * 60)
    print_roster(config)
    print("=" * 60)
    await arun_one_episode(
        env=env,
        agent_list=agents,
        omniscient=False,
        script_like=False,
        json_in_script=False,
        tag="test_rps",
        push_to_db=True,
    )


if __name__ == "__main__":
    # Configure logging for standalone execution
    LOG_FILE = BASE_DIR / "rps_game_debug.log"
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
