"""Launcher for the Stag Hunt game."""

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
    rank_split_rewards,
)
from sotopia.messages import AgentAction, Message, Observation, SimpleMessage
from sotopia.server import arun_one_episode

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"

os.environ.setdefault("REDIS_OM_URL", "redis://:@localhost:6379")
redis.Redis(host="localhost", port=6379)

_gen_logger = logging.getLogger("sotopia.generation")
_env_logger = logging.getLogger("sotopia.envs.social_game")


class StagHuntEvaluator(SocialGameEndEvaluator):
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


_NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
_ORDINALS = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th", 5: "5th", 6: "6th"}


def _tie_example(totals: List[int]) -> str:
    """What `rank_split_rewards` gives these totals, in words ("the player on 30 wins, ...")."""
    rewards = rank_split_rewards({str(i): t for i, t in enumerate(totals)})
    verbs = {1.0: ("wins", "win"), 0.0: ("draws", "draw"), -1.0: ("loses", "lose")}
    parts = []
    for total in sorted(set(totals), reverse=True):
        count = totals.count(total)
        verb = verbs[rewards[str(totals.index(total))]][0 if count == 1 else 1]
        who = "the player" if count == 1 else f"the {_NUMBER_WORDS[count]} players"
        parts.append(f"{who} on {total} {verb}")
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def tie_rule(n: int) -> str:
    """How `rank_split_rewards` treats equal totals at an n-player table, with two worked examples."""
    k = n // 2
    other = (
        f"one that holds {_ORDINALS[k + 1]} place or straddles the line between winners and losers"
        if n % 2
        else f"one that straddles the line between {_ORDINALS[k]} and {_ORDINALS[k + 1]} place"
    )
    a = [30] * (k + 1) + [12] * (n - k - 1)
    b = [30, 20, 20] + [12] * (n - 3)
    return (
        f"Players with equal totals share one result: a tied group that lies entirely within the top {k} places "
        f"wins, one that lies entirely within the bottom {k} places loses, and any other tied group ({other}) draws. "
        f"For example, with totals {', '.join(map(str, a))}, {_tie_example(a)}; with totals "
        f"{', '.join(map(str, b))}, {_tie_example(b)}."
    )


MOVES = ("stag", "hare")
# Singular and plural name the same hunt ("stags", "hares").
MOVE_WORDS = {"stag": ("stag", "stags"), "hare": ("hare", "hares")}


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
    """'stag' or 'hare' when the reply names exactly one of them as a word, else None.

    A negated move is not named, so "not hare, stag" is stag. A reply naming both ("hare is safe but I
    pick stag") or neither ("I won't hunt stag") is not read as a move; the player is asked again.
    """
    text = argument.lower().replace("’", "'")
    named = [move for move, words in MOVE_WORDS.items() if any(names(text, w) for w in words)]
    return named[0] if len(named) == 1 else None


class StagHuntActionHandler(ActionHandler):
    def handle_action(
        self, env: SocialDeductionGame, agent_name: str, action: AgentAction
    ) -> None:
        if not isinstance(env, StagHuntEnv):
            return
        if env.current_state != "Choose" or agent_name not in env.awaiting():
            return
        move = (
            read_move(action.argument)
            if action.action_type in ["action", "speak"]
            else None
        )
        env.take_reply(agent_name, move)

    def get_action_instruction(self, env: SocialDeductionGame, agent_name: str) -> str:
        if not isinstance(env, StagHuntEnv):
            return ""
        if env.current_state == "Choose":
            round_num = env.internal_state.get("round", 0) + 1
            max_rounds = env._config.get("max_rounds", 10)
            scores = env.internal_state.get("scores", {})
            text = (
                f"Round {round_num}/{max_rounds}. Scores: {scores}. "
                f"Choose 'stag' or 'hare'. "
                f"All stag=5 each. Any hare: stag=0, hare=2. "
                f"Players in the top half by total score win; players in the bottom half "
                f"lose. {tie_rule(len(env.agents))}"
            )
            if env.internal_state.get("unclear", {}).get(agent_name):
                text += (
                    " Your last reply did not name exactly one of 'stag' or 'hare', so you are asked again."
                    " If this reply is also unclear, your move is drawn at random."
                )
            return text
        return ""


class StagHuntEnv(SocialDeductionGame):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(action_handler=StagHuntActionHandler(), **kwargs)

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
            "current_moves": {},
            "unclear": {},
            "drawn": [],
            "game_over": False,
            "final_scores": {},
            "end_reason": "",
        }
        # Moves drawn for twice-unclear replies; seeded from the roster so replays agree.
        self._fallback_rng = random.Random(f"{self._config.get('seed')}-fallback")
        return obs

    def awaiting(self) -> List[str]:
        """Players whose move for the current round is not in yet (everyone, at the start of a round)."""
        moves = self.internal_state.get("current_moves", {})
        return [a for a in self.agents if a not in moves]

    def take_reply(self, agent_name: str, move: str | None) -> None:
        """Hold a clear move; ask an unclear (or failed) reply once more, then draw the move at random.

        Moves stay hidden until the round is complete: the Choose state is private, so a player
        asked again never sees another player's reply, and no reply text reaches the others.
        """
        if move is not None:
            self.internal_state["current_moves"][agent_name] = move
            return
        unclear = self.internal_state.setdefault("unclear", {})
        unclear[agent_name] = unclear.get(agent_name, 0) + 1
        round_num = self.internal_state.get("round", 0) + 1
        if unclear[agent_name] == 1:
            message = (
                f"[Round {round_num}] {agent_name}'s reply did not name exactly one of 'stag' or 'hare', "
                f"so {agent_name} is asked again. Moves stay hidden until the round is complete."
            )
        else:
            self.internal_state["current_moves"][agent_name] = (
                self._fallback_rng.choice(MOVES)
            )
            self.internal_state.setdefault("drawn", []).append(agent_name)
            message = (
                f"[Round {round_num}] {agent_name}'s second reply was also unclear, "
                f"so {agent_name}'s move is drawn at random."
            )
        self.recv_message("Environment", SimpleMessage(message=message))

    def _update_action_mask(self) -> None:
        super()._update_action_mask()
        if self.current_state == "Choose" and "current_moves" in self.internal_state:
            due = self.awaiting()
            self.action_mask = [a in due for a in self.agents]

    def _check_eliminations(self) -> None:
        if self.current_state == "Choose" and not self.awaiting():
            moves = self.internal_state["current_moves"]
            drawn = self.internal_state.get("drawn", [])
            scores = self.internal_state["scores"]
            all_stag = all(m == "stag" for m in moves.values())

            for name, move in moves.items():
                if all_stag:
                    scores[name] += 5
                elif move == "hare":
                    scores[name] += 2

            round_num = self.internal_state["round"] + 1
            self.internal_state["round"] = round_num
            self.internal_state["current_moves"] = {}
            self.internal_state["unclear"] = {}
            self.internal_state["drawn"] = []

            move_str = ", ".join(
                f"{n}: {moves[n]}" + (" (drawn at random)" if n in drawn else "")
                for n in self.agents
            )
            self.recv_message(
                "Environment",
                SimpleMessage(
                    message=f"[Round {round_num}] Choices: {move_str}. "
                    f"{'All chose stag! 5 pts each!' if all_stag else 'Not all stag.'} "
                    f"Scores: {scores}"
                ),
            )

            max_rounds = self._config.get("max_rounds", 10)
            if round_num >= max_rounds:
                final = rank_split_rewards(scores)
                reason = f"Game over after {max_rounds} rounds. Final scores: {scores}"

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
) -> StagHuntEnv:
    return StagHuntEnv(
        env_profile=env_profile,
        config=config,
        model_name=model_name,
        # Safety cap only: every round takes one step, or two when a player is asked again.
        evaluators=[
            StagHuntEvaluator(max_turn_number=2 * config.get("max_rounds", 10))
        ],
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
    scenario = config.get("description", "Stag Hunt").format(
        agent_names=", ".join(agent_names)
    )
    env_profile = EnvironmentProfile(
        scenario=scenario,
        relationship=RelationshipType.acquaintance,
        agent_goals=agent_goals,
        tag="stag_hunt",
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

    parser = argparse.ArgumentParser(description="Run Stag Hunt game.")
    parser.add_argument("--config", type=str, default=str(CONFIG_PATH))
    parser.add_argument("--roster", type=str, default=str(BASE_DIR / "roster.json"))
    args = parser.parse_args()
    config = load_config(args.config)
    roster = load_config(args.roster)
    config["agents"] = roster.get("agents", [])
    agent_model_name = get_model_names(config)
    env, agents = prepare_scenario("gpt-4o", agent_model_name, config)
    print("Stag Hunt")
    print("=" * 60)
    print_roster(config)
    print("=" * 60)
    await arun_one_episode(
        env=env,
        agent_list=agents,
        omniscient=False,
        script_like=False,
        json_in_script=False,
        tag="test_stag_hunt",
        push_to_db=True,
    )


if __name__ == "__main__":
    LOG_FILE = BASE_DIR / "stag_hunt_game_debug.log"
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
