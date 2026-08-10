"""Launcher for the Insider social game scenario."""

from __future__ import annotations

import asyncio
import logging
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import redis
from litellm import completion as litellm_completion
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


class InsiderEvaluator(SocialGameEndEvaluator):
    """Evaluator for Insider win conditions."""

    def __call__(
        self, turn_number: int, messages: List[Tuple[str, Message]], **kwargs: Any
    ) -> List[Tuple[str, Tuple[Tuple[str, int | float | bool], str]]]:
        if turn_number >= self.max_turn_number:
            env = kwargs.get("env")
            if env:
                # Timeout = word not guessed = everyone loses
                response: List[
                    Tuple[str, Tuple[Tuple[str, int | float | bool], str]]
                ] = [
                    (
                        "environment",
                        (("terminated", True), "Max turns reached. Word not guessed."),
                    )
                ]
                for idx, name in enumerate(env.agents):
                    response.append(
                        (f"agent_{idx + 1}", (("complete_rating", -1.0), "Timeout"))
                    )
                return response
            return [("environment", (("terminated", True), "Max turns reached."))]

        env = kwargs.get("env")
        if not env:
            return [("environment", (("terminated", False), ""))]

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


class InsiderActionHandler(ActionHandler):
    """Handles actions for Insider game."""

    def handle_action(
        self, env: SocialDeductionGame, agent_name: str, action: AgentAction
    ) -> None:
        if not isinstance(env, InsiderEnv):
            return

        if action.action_type not in ["action", "speak"]:
            return

        if env.current_state == "Questioning":
            # During questioning, agents ask yes/no questions via speak
            if action.action_type == "speak":
                question = action.argument.strip()
                if question:
                    env.internal_state["questions_asked"] = (
                        env.internal_state.get("questions_asked", 0) + 1
                    )
                    # Answer the question using fact sheet
                    answer = self._answer_question(env, question)
                    env.recv_message(
                        "Environment",
                        SimpleMessage(message=f"[Answer] {answer}"),
                    )

                    # Check if question limit reached
                    max_q = env._config.get("max_questions", 15)
                    asked = env.internal_state["questions_asked"]
                    if asked >= max_q:
                        env.recv_message(
                            "Environment",
                            SimpleMessage(
                                message=f"[Game] Question limit ({max_q}) reached! "
                                f"Time to guess the word."
                            ),
                        )

        elif env.current_state == "Guess":
            arg = action.argument.strip().lower()
            secret_word = env.internal_state.get("secret_word", "").lower()

            if arg.startswith("pass") or arg.startswith("skip"):
                return

            # Extract guess word
            guess_word = arg.replace("guess", "").strip()
            if not guess_word:
                guess_word = arg

            if guess_word == secret_word:
                env.internal_state["word_guessed"] = True
                env.internal_state["guesser"] = agent_name
                env.recv_message(
                    "Environment",
                    SimpleMessage(
                        message=f"[Game] {agent_name} correctly guessed the word: {secret_word}! "
                        f"Now discuss who you think the Insider is."
                    ),
                )
            else:
                if "incorrect_guesses" not in env.internal_state:
                    env.internal_state["incorrect_guesses"] = []
                env.internal_state["incorrect_guesses"].append((agent_name, guess_word))

        elif env.current_state == "Vote":
            arg = action.argument.strip().lower()
            # Parse "vote NAME"
            target = None
            words = action.argument.split()
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

    def _answer_question(self, env: InsiderEnv, question: str) -> str:
        """Answer a yes/no question about the secret word via an LLM judge.

        Uses a small model (default gpt-4o-mini) primed with the secret word.
        The judge sees only the secret word and the question -- it has no
        access to other agents' messages or the game state.
        """
        secret_word = env.internal_state.get(
            "secret_word", env._config.get("secret_word", "")
        )
        if not secret_word:
            return "I can only answer yes or no. Try to be more specific."

        judge_model = env._config.get("judge_model", "gpt-4o-mini")
        prompt = (
            f"The secret word is: {secret_word}.\n"
            f"Answer the following yes/no question about the word with exactly "
            f"one word: Yes, No, or Unclear (if the question is not a "
            f"well-formed yes/no question or cannot be answered from common "
            f"knowledge about the word).\n\n"
            f"Question: {question.strip()}"
        )

        try:
            resp = litellm_completion(
                model=judge_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4,
                temperature=0.0,
            )
            reply = (resp.choices[0].message.content or "").strip().lower()
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "Insider judge call failed (%s); falling back.", exc
            )
            return "I can only answer yes or no. Try to be more specific."

        if reply.startswith("yes"):
            return "Yes"
        if reply.startswith("no"):
            return "No"
        return "I can only answer yes or no. Try to be more specific."

    def get_action_instruction(self, env: SocialDeductionGame, agent_name: str) -> str:
        if not isinstance(env, InsiderEnv):
            return ""

        if env.current_state == "Questioning":
            asked = env.internal_state.get("questions_asked", 0)
            max_q = env._config.get("max_questions", 15)
            category = env.internal_state.get(
                "secret_word_category", env._config.get("secret_word_category", "")
            )
            hint = f" The category is: {category}." if category else ""
            return (
                f"Questioning phase ({asked}/{max_q} questions asked).{hint} "
                f"Ask a yes/no question to help figure out the secret word. "
                f"The environment will answer truthfully."
            )
        elif env.current_state == "Guess":
            if env.internal_state.get("word_guessed", False):
                return "The word has been guessed! Waiting for others."
            return (
                "Time to guess the secret word! Use 'guess WORD' to guess, "
                "or 'pass' if you don't know."
            )
        elif env.current_state == "Discussion":
            return (
                "The word was guessed! Now discuss who you think the Insider is. "
                "Think about which questions seemed suspiciously helpful or insightful."
            )
        elif env.current_state == "Vote":
            player_list = ", ".join(list(env.agents))
            return (
                f"Vote for who you think is the Insider. Players: [{player_list}]. "
                f"Use 'vote NAME' (e.g., 'vote Alice')."
            )
        return ""


# ============================================================================
# Environment
# ============================================================================


class InsiderEnv(SocialDeductionGame):
    """Insider game with questioning, guessing, and voting."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(action_handler=InsiderActionHandler(), **kwargs)

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

        # Draw a secret word for this episode.
        #
        # Precedence for repeatability:
        #   1. options/config explicit override: "secret_word" (+ optional "secret_word_category")
        #      lets a roster pin a specific word.
        #   2. else, sample deterministically from "secret_word_pool" using the env seed.
        #      The pool is sorted by word before sampling so the seed -> word mapping
        #      is stable when new words are appended to the pool.
        explicit_word = ""
        if options:
            explicit_word = options.get("secret_word", "")
        if not explicit_word:
            explicit_word = self._config.get("secret_word", "")

        pool = self._config.get("secret_word_pool", [])
        # The tournament runner calls reset() with seed=None; fall back to
        # config["seed"] (populated from the roster) for reproducibility.
        effective_seed = seed if seed is not None else self._config.get("seed")
        rng = random.Random(effective_seed)
        if explicit_word:
            secret_word = explicit_word
            # Try to recover category from pool, else from config
            cat_lookup = {e.get("word", ""): e.get("category", "") for e in pool}
            category = cat_lookup.get(
                explicit_word, self._config.get("secret_word_category", "")
            )
        elif pool:
            sorted_pool = sorted(pool, key=lambda e: e.get("word", ""))
            entry = rng.choice(sorted_pool)
            secret_word = entry.get("word", "")
            category = entry.get("category", "")
        else:
            secret_word = ""
            category = self._config.get("secret_word_category", "")

        self.internal_state = {
            "questions_asked": 0,
            "word_guessed": False,
            "guesser": None,
            "incorrect_guesses": [],
            "votes": {},
            "game_over": False,
            "final_scores": {},
            "end_reason": "",
            "secret_word": secret_word,
            "secret_word_category": category,
        }

        # Announce category
        if category:
            self.recv_message(
                "Environment",
                SimpleMessage(
                    message=f"[Game] The secret word category is: {category}. "
                    f"Ask yes/no questions to figure out the word!"
                ),
            )

        # Privately tell the Insider the word
        for agent_name in self.agents:
            role = self.agent_to_role.get(agent_name, "")
            if role == "Insider":
                self.recv_message(
                    "Environment",
                    SimpleMessage(
                        message=f"[Private] You are the Insider. The secret word is: {secret_word}. "
                        f"Subtly guide the group to guess it without being too obvious."
                    ),
                    receivers=[agent_name],
                )

        return obs

    def _check_eliminations(self) -> None:
        """Handle state transitions based on game progress."""
        if not self._should_transition_state():
            return

        # Check if question limit reached -> transition to Guess
        if self.current_state == "Questioning":
            asked = self.internal_state.get("questions_asked", 0)
            max_q = self._config.get("max_questions", 15)
            if asked >= max_q:
                self.recv_message(
                    "Environment",
                    SimpleMessage(
                        message=f"[Game] All {max_q} questions have been asked! Time to guess the word."
                    ),
                )
                self.current_state = "Guess"
                if hasattr(self, "_state_turn_count"):
                    self._state_turn_count["Guess"] = 0
                if hasattr(self, "_round_robin_idx"):
                    self._round_robin_idx = 0

        elif self.current_state == "Guess":
            if not self.internal_state.get("word_guessed", False):
                # Word not guessed - everyone loses
                scores = {name: -1.0 for name in self.agents}
                secret_word = self.internal_state.get("secret_word", "")
                self.internal_state["game_over"] = True
                self.internal_state["final_scores"] = scores
                self.internal_state["end_reason"] = (
                    f"No one guessed the word ({secret_word}). Everyone loses!"
                )
                self.recv_message(
                    "Environment",
                    SimpleMessage(
                        message=f"[Game] No one guessed the word. The word was: {secret_word}. Everyone loses!"
                    ),
                )

        elif self.current_state == "Vote":
            votes = self.internal_state.get("votes", {})
            if votes:
                vote_counts = Counter(votes.values())
                # Find most voted player
                most_voted = vote_counts.most_common(1)[0][0]
                vote_count = vote_counts[most_voted]

                # Find the actual Insider
                insider_name = None
                for name in self.agents:
                    if self.agent_to_role.get(name, "") == "Insider":
                        insider_name = name
                        break

                scores = {}
                if most_voted == insider_name:
                    # Insider caught!
                    reason = (
                        f"The group correctly identified {insider_name} as the Insider "
                        f"({vote_count} votes)! Citizens win!"
                    )
                    citizen_team = [
                        n
                        for n in self.agents
                        if self.agent_to_role.get(n, "") != "Insider"
                    ]
                    insider_team = [
                        n
                        for n in self.agents
                        if self.agent_to_role.get(n, "") == "Insider"
                    ]
                    n_winners = len(citizen_team)
                    n_losers = len(insider_team)
                    for name in citizen_team:
                        scores[name] = 1.0 / n_winners
                    for name in insider_team:
                        scores[name] = -1.0 / n_losers if n_losers > 0 else 0.0
                else:
                    # Insider escaped!
                    reason = (
                        f"The group voted for {most_voted} ({vote_count} votes), "
                        f"but the Insider was {insider_name}! Insider wins!"
                    )
                    insider_team = [
                        n
                        for n in self.agents
                        if self.agent_to_role.get(n, "") == "Insider"
                    ]
                    citizen_team = [
                        n
                        for n in self.agents
                        if self.agent_to_role.get(n, "") != "Insider"
                    ]
                    n_winners = len(insider_team)
                    n_losers = len(citizen_team)
                    for name in insider_team:
                        scores[name] = 1.0 / n_winners
                    for name in citizen_team:
                        scores[name] = -1.0 / n_losers if n_losers > 0 else 0.0

                secret_word = self.internal_state.get("secret_word", "")
                if secret_word and secret_word.lower() not in reason.lower():
                    reason = f"{reason} (secret word: {secret_word})"
                self.internal_state["game_over"] = True
                self.internal_state["final_scores"] = scores
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
) -> InsiderEnv:
    return InsiderEnv(
        env_profile=env_profile,
        config=config,
        model_name=model_name,
        evaluators=[InsiderEvaluator(max_turn_number=60)],
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

        # Fill in secret word for Insider
        secret_word = config.get("secret_word", "")
        if secrets and "{secret_word}" in secrets:
            secrets = secrets.replace("{secret_word}", secret_word)

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
    scenario = config.get("description", "Insider").format(
        agent_names=", ".join(agent_names)
    )
    env_profile = EnvironmentProfile(
        scenario=scenario,
        relationship=RelationshipType.acquaintance,
        agent_goals=agent_goals,
        tag="insider",
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

    parser = argparse.ArgumentParser(description="Run Insider game.")
    parser.add_argument("--config", type=str, default=str(CONFIG_PATH))
    parser.add_argument("--roster", type=str, default=str(BASE_DIR / "roster.json"))
    args = parser.parse_args()

    config = load_config(args.config)
    roster = load_config(args.roster)
    config["agents"] = roster.get("agents", [])

    agent_model_name = get_model_names(config)
    env, agents = prepare_scenario("gpt-4o", agent_model_name, config)

    print("Insider")
    print("=" * 60)
    print_roster(config)
    print("=" * 60)

    await arun_one_episode(
        env=env,
        agent_list=agents,
        omniscient=False,
        script_like=False,
        json_in_script=False,
        tag="test_insider",
        push_to_db=True,
    )


if __name__ == "__main__":
    LOG_FILE = BASE_DIR / "insider_game_debug.log"
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
