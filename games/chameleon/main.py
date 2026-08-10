"""Launcher for The Chameleon social game scenario."""

from __future__ import annotations

import asyncio
import logging
import os
import random
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


class ChameleonEvaluator(SocialGameEndEvaluator):
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


class ChameleonActionHandler(ActionHandler):
    def handle_action(
        self, env: SocialDeductionGame, agent_name: str, action: AgentAction
    ) -> None:
        if not isinstance(env, ChameleonEnv):
            return
        if action.action_type not in ["action", "speak"]:
            return

        if env.current_state == "Clue":
            clue = action.argument.strip().split()[0] if action.argument.strip() else ""
            if clue:
                if "clues" not in env.internal_state:
                    env.internal_state["clues"] = {}
                env.internal_state["clues"][agent_name] = clue

        elif env.current_state == "Vote":
            words = action.argument.split()
            target = None
            for w in words:
                for name in env.agents:
                    if w.lower() == name.lower():
                        target = name
                        break
                if target:
                    break
            if target:
                if "votes" not in env.internal_state:
                    env.internal_state["votes"] = {}
                env.internal_state["votes"][agent_name] = target

        elif env.current_state == "Guess":
            chameleon_name = env.internal_state.get("chameleon_name")
            if agent_name != chameleon_name:
                return
            arg = action.argument.strip().lower()
            guess_word = arg.replace("guess", "").strip()
            if not guess_word:
                guess_word = arg
            env.internal_state["chameleon_guess"] = guess_word

    def get_action_instruction(self, env: SocialDeductionGame, agent_name: str) -> str:
        if not isinstance(env, ChameleonEnv):
            return ""

        if env.current_state == "Clue":
            clues = env.internal_state.get("clues", {})
            given = [f"{n}: {c}" for n, c in clues.items()]
            clue_str = f" Clues so far: {', '.join(given)}." if given else ""
            return (
                f"Give a ONE-WORD clue related to the secret word.{clue_str} "
                f"If you are a Citizen, prove you know the word without making it too obvious. "
                f"If you are the Chameleon, give a plausible clue to blend in."
            )
        elif env.current_state == "Discussion":
            clues = env.internal_state.get("clues", {})
            clue_str = ", ".join(f"{n}: {c}" for n, c in clues.items())
            return (
                f"All clues: [{clue_str}]. "
                f"Discuss who you think the Chameleon is based on these clues."
            )
        elif env.current_state == "Vote":
            player_list = ", ".join(list(env.agents))
            return (
                f"Vote for who you think is the Chameleon. Players: [{player_list}]. "
                f"Use 'vote NAME' (e.g., 'vote Alice')."
            )
        elif env.current_state == "Guess":
            chameleon_name = env.internal_state.get("chameleon_name")
            if agent_name == chameleon_name:
                clues = env.internal_state.get("clues", {})
                clue_str = ", ".join(f"{n}: {c}" for n, c in clues.items())
                word_list = env._config.get("word_list", "")
                return (
                    f"You were identified as the Chameleon! You get one last chance. "
                    f"All clues were: [{clue_str}]. "
                    f"Possible words: [{word_list}]. "
                    f"Use 'guess WORD' to guess the secret word."
                )
            else:
                return "Waiting for the Chameleon to guess the secret word."
        return ""


# ============================================================================
# Environment
# ============================================================================


class ChameleonEnv(SocialDeductionGame):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(action_handler=ChameleonActionHandler(), **kwargs)

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

        # Find the Chameleon
        chameleon_name = None
        for name in self.agents:
            if self.agent_to_role.get(name, "") == "Chameleon":
                chameleon_name = name
                break

        self.internal_state = {
            "clues": {},
            "votes": {},
            "chameleon_name": chameleon_name,
            "chameleon_guess": None,
            "game_over": False,
            "final_scores": {},
            "end_reason": "",
        }

        # Send private info to each player
        category = self._config.get("category", "")
        secret_word = self._config.get("secret_word", "")
        word_list = self._config.get("word_list", "")

        for agent_name in self.agents:
            role = self.agent_to_role.get(agent_name, "")
            if role == "Chameleon":
                self.recv_message(
                    "Environment",
                    SimpleMessage(
                        message=f"[Private] You are the Chameleon. "
                        f"The category is: {category}. "
                        f"You do NOT know the secret word. "
                        f"The possible words are: [{word_list}]. "
                        f"Give a plausible clue to blend in."
                    ),
                    receivers=[agent_name],
                )
            else:
                self.recv_message(
                    "Environment",
                    SimpleMessage(
                        message=f"[Private] You are a Citizen. "
                        f"The category is: {category}. "
                        f"The secret word is: {secret_word}."
                    ),
                    receivers=[agent_name],
                )

        return obs

    def _update_action_mask(self) -> None:
        """In Guess state, only the Chameleon acts."""
        super()._update_action_mask()
        if self.current_state == "Guess":
            chameleon_name = self.internal_state.get("chameleon_name")
            agents = list(self.agents)
            for idx, name in enumerate(agents):
                self.action_mask[idx] = name == chameleon_name

    def _check_eliminations(self) -> None:
        if not self._should_transition_state():
            return

        if self.current_state == "Vote":
            votes = self.internal_state.get("votes", {})
            if not votes:
                return

            vote_counts = Counter(votes.values())
            most_voted = vote_counts.most_common(1)[0][0]
            vote_count = vote_counts[most_voted]
            chameleon_name = self.internal_state.get("chameleon_name")

            vote_summary = ", ".join(f"{v}: {k}" for k, v in votes.items())
            self.recv_message(
                "Environment",
                SimpleMessage(
                    message=f"[Vote Results] {vote_summary}. "
                    f"{most_voted} received {vote_count} votes."
                ),
            )

            if most_voted != chameleon_name:
                # Wrong person voted -- Chameleon wins immediately
                chameleon_team = [
                    n
                    for n in self.agents
                    if self.agent_to_role.get(n, "") == "Chameleon"
                ]
                citizen_team = [
                    n
                    for n in self.agents
                    if self.agent_to_role.get(n, "") != "Chameleon"
                ]
                n_winners = len(chameleon_team)
                n_losers = len(citizen_team)
                scores = {}
                for name in chameleon_team:
                    scores[name] = 1.0 / n_winners
                for name in citizen_team:
                    scores[name] = -1.0 / n_losers if n_losers > 0 else 0.0

                reason = (
                    f"The group voted for {most_voted}, but the Chameleon was "
                    f"{chameleon_name}! Chameleon wins!"
                )
                self.internal_state["game_over"] = True
                self.internal_state["final_scores"] = scores
                self.internal_state["end_reason"] = reason
                self.recv_message(
                    "Environment", SimpleMessage(message=f"[Game] {reason}")
                )
            else:
                # Correct! Chameleon caught -- but gets a chance to guess
                self.recv_message(
                    "Environment",
                    SimpleMessage(
                        message=f"[Game] The group correctly identified {chameleon_name} "
                        f"as the Chameleon! But the Chameleon gets one last chance "
                        f"to guess the secret word..."
                    ),
                )

        elif self.current_state == "Guess":
            guess = self.internal_state.get("chameleon_guess")
            secret_word = self._config.get("secret_word", "").lower()
            chameleon_name = self.internal_state.get("chameleon_name")
            scores: Dict[str, float] = {}

            # Default-and-continue: if the caught Chameleon failed to submit
            # a parseable guess, treat it as a wrong guess so the game ends
            # instead of stalling in the Guess state until max_turn timeout.
            if guess is None:
                guess = ""

            if guess == secret_word:
                # Chameleon guessed correctly -- Chameleon wins
                reason = (
                    f"{chameleon_name} correctly guessed the word '{secret_word}'! "
                    f"Chameleon wins despite being caught!"
                )
                chameleon_team = [
                    n
                    for n in self.agents
                    if self.agent_to_role.get(n, "") == "Chameleon"
                ]
                citizen_team = [
                    n
                    for n in self.agents
                    if self.agent_to_role.get(n, "") != "Chameleon"
                ]
                n_winners = len(chameleon_team)
                n_losers = len(citizen_team)
                for name in chameleon_team:
                    scores[name] = 1.0 / n_winners
                for name in citizen_team:
                    scores[name] = -1.0 / n_losers if n_losers > 0 else 0.0
            else:
                # Chameleon guessed wrong -- Citizens win
                reason = (
                    f"{chameleon_name} guessed '{guess}' but the word was "
                    f"'{secret_word}'. Citizens win!"
                )
                chameleon_team = [
                    n
                    for n in self.agents
                    if self.agent_to_role.get(n, "") == "Chameleon"
                ]
                citizen_team = [
                    n
                    for n in self.agents
                    if self.agent_to_role.get(n, "") != "Chameleon"
                ]
                n_winners = len(citizen_team)
                n_losers = len(chameleon_team)
                for name in citizen_team:
                    scores[name] = 1.0 / n_winners
                for name in chameleon_team:
                    scores[name] = -1.0 / n_losers if n_losers > 0 else 0.0

            self.internal_state["game_over"] = True
            self.internal_state["final_scores"] = scores
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
) -> ChameleonEnv:
    return ChameleonEnv(
        env_profile=env_profile,
        config=config,
        model_name=model_name,
        evaluators=[ChameleonEvaluator(max_turn_number=25)],
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

        # Fill in template variables
        category = config.get("category", "")
        secret_word = config.get("secret_word", "")
        word_list = config.get("word_list", "")
        if secrets:
            secrets = secrets.replace("{category}", category)
            secrets = secrets.replace("{secret_word}", secret_word)
            secrets = secrets.replace("{word_list}", word_list)

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


def _pin_secret_word_from_pool(config: Dict[str, Any]) -> None:
    """Pick a category + secret word from the configured pool.

    Deterministic from config["seed"] so the same roster always gets the same
    word. Skipped if config already has an explicit secret_word pinned, or if
    no category_pool is configured (legacy single-word setup).
    """
    if config.get("secret_word"):
        return
    pool = config.get("category_pool", [])
    if not pool:
        return
    rng = random.Random(config.get("seed"))
    sorted_pool = sorted(pool, key=lambda c: c.get("category", ""))
    chosen = rng.choice(sorted_pool)
    config["category"] = chosen.get("category", "")
    sorted_words = sorted(chosen.get("word_list", []))
    config["secret_word"] = rng.choice(sorted_words) if sorted_words else ""
    config["word_list"] = ", ".join(chosen.get("word_list", []))


def prepare_scenario(
    env_model_name: str,
    agent_model_name: str | Dict[str, str] | List[str],
    config: Dict[str, Any] | None = None,
) -> tuple[SocialDeductionGame, List[LLMAgent]]:
    if config is None:
        config = load_config(CONFIG_PATH)
    _pin_secret_word_from_pool(config)
    agent_profiles = [ensure_agent_profile(entry) for entry in config.get("agents", [])]
    agent_goals = [
        config.get("role_goals", {}).get(entry.get("role", ""), "")
        for entry in config.get("agents", [])
    ]
    agent_names = [entry.get("name", "") for entry in config.get("agents", [])]
    scenario = config.get("description", "The Chameleon").format(
        agent_names=", ".join(agent_names)
    )
    env_profile = EnvironmentProfile(
        scenario=scenario,
        relationship=RelationshipType.acquaintance,
        agent_goals=agent_goals,
        tag="chameleon",
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

    parser = argparse.ArgumentParser(description="Run The Chameleon game.")
    parser.add_argument("--config", type=str, default=str(CONFIG_PATH))
    parser.add_argument("--roster", type=str, default=str(BASE_DIR / "roster.json"))
    args = parser.parse_args()

    config = load_config(args.config)
    roster = load_config(args.roster)
    config["agents"] = roster.get("agents", [])

    agent_model_name = get_model_names(config)
    env, agents = prepare_scenario("gpt-4o", agent_model_name, config)

    print("The Chameleon")
    print("=" * 60)
    print_roster(config)
    print("=" * 60)

    await arun_one_episode(
        env=env,
        agent_list=agents,
        omniscient=False,
        script_like=False,
        json_in_script=False,
        tag="test_chameleon",
        push_to_db=True,
    )


if __name__ == "__main__":
    LOG_FILE = BASE_DIR / "chameleon_game_debug.log"
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
