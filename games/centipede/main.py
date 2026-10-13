"""Launcher for the Centipede Game."""

from __future__ import annotations

import asyncio
import os
import logging
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

_gen_logger = logging.getLogger("sotopia.generation")
_env_logger = logging.getLogger("sotopia.envs.social_game")


def turn_limit(config: Dict[str, Any]) -> int:
    """Env steps a match may take: every decision of every game, each asked at most twice.

    One unclear reply is asked again and a second one is replaced by a random move, so a match
    can never need more; the limit only guards against a stalled engine.
    """
    return 2 * config.get("num_games", 4) * len(config.get("nodes", []))


class CentipedeEvaluator(SocialGameEndEvaluator):
    def __call__(
        self, turn_number: int, messages: List[Tuple[str, Message]], **kwargs: Any
    ) -> List[Tuple[str, Tuple[Tuple[str, int | float | bool], str]]]:
        env = kwargs.get("env")
        if (
            env
            and not env.internal_state.get("game_over", False)
            and turn_number >= self.max_turn_number
        ):
            # The turn limit ends the match as its last game would: the higher cumulative score wins.
            env.finish_match(f"Turn limit of {self.max_turn_number} reached.")
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
    """'take' or 'pass' when the reply names exactly one of them as a word, else None.

    A negated move is not named, so "I will not take, I pass" is pass. A reply naming both ("take or pass")
    or neither ("I won't take") is not read as a move; the engine says so publicly and the same player
    chooses again.
    """
    text = argument.lower().replace("’", "'")
    said = [m for m in ("take", "pass") if names(text, m)]
    return said[0] if len(said) == 1 else None


class CentipedeActionHandler(ActionHandler):
    def handle_action(
        self, env: SocialDeductionGame, agent_name: str, action: AgentAction
    ) -> None:
        if not isinstance(env, CentipedeEnv):
            return
        if env.current_state != "Turn":
            return
        agents = list(env.agents)
        current_idx = env.internal_state.get("current_player_idx", 0)
        if agent_name != agents[current_idx]:
            return
        move = read_move(action.argument) if action.action_type == "action" else None
        if move:
            env.internal_state["decision"] = move
            env.internal_state["unclear"] = 0
            return
        game_num = env.internal_state.get("game_num", 0) + 1
        unclear = env.internal_state.get("unclear", 0) + 1
        env.internal_state["unclear"] = unclear
        # The announcement never quotes the reply: the players cannot talk, so only parsed moves are shown.
        if unclear == 1:
            message = (
                f"[Game {game_num}] {agent_name}'s reply did not name exactly one of 'take' or 'pass', "
                f"so {agent_name} is asked once more."
            )
        else:
            # Asked once already: the move is drawn at random, take and pass equally likely.
            drawn = env._fallback_rng.choice(("take", "pass"))
            env.internal_state["decision"] = drawn
            env.internal_state["unclear"] = 0
            message = (
                f"[Game {game_num}] {agent_name}'s second reply was also unclear, "
                f"so {agent_name}'s move is drawn at random: {drawn}."
            )
        env.recv_message("Environment", SimpleMessage(message=message))

    def get_action_instruction(self, env: SocialDeductionGame, agent_name: str) -> str:
        if not isinstance(env, CentipedeEnv):
            return ""
        if env.current_state == "Turn":
            agents = list(env.agents)
            current_idx = env.internal_state.get("current_player_idx", 0)
            current_player = agents[current_idx]
            node = env.internal_state.get("current_node", 0)
            game_idx = env.internal_state.get("game_num", 0)
            num_games = env._config.get("num_games", 4)
            nodes = env._config.get("nodes", [])
            scores = env.internal_state.get("scores", {})

            if agent_name != current_player:
                return (
                    f"Game {game_idx + 1}/{num_games}. "
                    f"Waiting for {current_player} to decide. "
                    f"Cumulative scores: {scores}"
                )

            actor = agents[current_idx]
            other = agents[1 - current_idx]
            first, _ = env.movers(game_idx)
            role = "first" if actor == first else "second"
            now = env.shares(game_idx, node)
            # The rest of the tree from here, in the mover's own terms (complete information).
            tree = []
            for k in range(node, len(nodes)):
                p = env.shares(game_idx, k)
                mover = "you move" if (k - node) % 2 == 0 else f"{other} moves"
                tree.append(f"node {k + 1} ({mover}): ({p[actor]}, {p[other]})")
            through = env.shares(game_idx, len(nodes))
            tree.append(f"if node {len(nodes)} is passed: ({through[actor]}, {through[other]})")

            text = (
                f"Game {game_idx + 1}/{num_games}, Node {node + 1}/{len(nodes)}. "
                f"You are the {role} mover in this game. Your turn: choose 'take' or 'pass'. "
                f"Take now: you ({actor}) get {now[actor]}, {other} gets {now[other]}. "
                f"Pass: the game goes on and {other} chooses at the next node. "
                f"Payoffs (you, {other}) if the game ends at each remaining node: "
                + "; ".join(tree)
                + f". Cumulative scores: {scores}"
            )
            if env.internal_state.get("unclear"):
                text += (
                    ". Your last reply was not a clear 'take' or 'pass'. "
                    "If this reply is also unclear, your move is drawn at random."
                )
            return text
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
            "unclear": 0,
            "scores": {name: 0 for name in self.agents},
            "game_over": False,
            "final_scores": {},
            "end_reason": "",
        }
        # Moves drawn for twice-unclear replies; seeded from the roster so replays agree.
        self._fallback_rng = random.Random(f"{self._config.get('seed')}-fallback")
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

    def movers(self, game_idx: int) -> Tuple[str, str]:
        """(first mover, second mover) of game `game_idx` (0-based); the first mover alternates by seat."""
        agents = list(self.agents)
        return agents[game_idx % 2], agents[1 - game_idx % 2]

    def shares(self, game_idx: int, node: int) -> Dict[str, int]:
        """Points each player gets if game `game_idx` ends at `node` (0-based; node == len(nodes) is the pass-through).

        The config's pairs are (first mover, second mover) of that game, the textbook centipede reading
        (McKelvey and Palfrey 1992; description.md's table): whoever takes gets the larger share of the node.
        """
        nodes = self._config.get("nodes", [])
        pair = nodes[node] if node < len(nodes) else self._config.get("pass_through_payoff", [0, 0])
        first, second = self.movers(game_idx)
        return {first: pair[0], second: pair[1]}

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

        # Each node's pair is (first mover, second mover) of this game, whoever moves at the node.
        actor = agents[current_idx]
        other = agents[1 - current_idx]

        if decision == "take":
            payoffs = self.shares(game_num, node)
            scores[actor] += payoffs[actor]
            scores[other] += payoffs[other]

            self.recv_message(
                "Environment",
                SimpleMessage(
                    message=f"[Game {game_num + 1}] {actor} TAKES at node {node + 1}! "
                    f"Payoffs: {actor}={payoffs[actor]}, {other}={payoffs[other]}. "
                    f"Cumulative scores: {scores}"
                ),
            )
            self._start_next_game_or_end()

        elif decision == "pass":
            next_node = node + 1
            if next_node >= len(nodes):
                payoffs = self.shares(game_num, next_node)
                scores[actor] += payoffs[actor]
                scores[other] += payoffs[other]

                self.recv_message(
                    "Environment",
                    SimpleMessage(
                        message=f"[Game {game_num + 1}] {actor} passes at final node! "
                        f"Full cooperation! Payoffs: {actor}={payoffs[actor]}, {other}={payoffs[other]}. "
                        f"Cumulative scores: {scores}"
                    ),
                )
                self._start_next_game_or_end()
            else:
                self.internal_state["current_node"] = next_node
                self.internal_state["current_player_idx"] = 1 - current_idx
                self.internal_state["decision"] = None

                next_player = other
                next_payoffs = self.shares(game_num, next_node)
                self.recv_message(
                    "Environment",
                    SimpleMessage(
                        message=f"[Game {game_num + 1}] {actor} passes! Pot grows. "
                        f"Now {next_player}'s turn at node {next_node + 1}. "
                        f"Take payoffs: {next_player}={next_payoffs[next_player]}, {actor}={next_payoffs[actor]}"
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
            self.finish_match(f"All {num_games} games done.")
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

    def finish_match(self, head: str) -> None:
        """End the match on the cumulative scores so far: higher wins (+1 / -1), equal is a tie (0)."""
        agents = list(self.agents)
        scores = self.internal_state["scores"]
        s1, s2 = scores[agents[0]], scores[agents[1]]
        if s1 > s2:
            final = {agents[0]: 1.0, agents[1]: -1.0}
            reason = f"{head} {agents[0]} wins! Final: {scores}"
        elif s2 > s1:
            final = {agents[0]: -1.0, agents[1]: 1.0}
            reason = f"{head} {agents[1]} wins! Final: {scores}"
        else:
            final = {agents[0]: 0.0, agents[1]: 0.0}
            reason = f"{head} Tie! Final: {scores}"

        self.internal_state["game_over"] = True
        self.internal_state["final_scores"] = final
        self.internal_state["end_reason"] = reason
        self.recv_message("Environment", SimpleMessage(message=f"[Game] {reason}"))


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
        evaluators=[CentipedeEvaluator(max_turn_number=turn_limit(config))],
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
