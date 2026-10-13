"""Launcher for The Chameleon social game scenario."""

from __future__ import annotations

import asyncio
import logging
import os
import random
import re
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
# Reading replies
# ============================================================================

_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})
# Between a command word and the name it takes: punctuation, markdown, quotes, "for" ("vote for Bob").
_GAP = r"""[\s:*_'"`@(\[-]*(?:(?:for|against|out)\s+[\s*_'"`@(\[-]*)?"""
_VOTE_VERB = r"vot(?:e|es|ed|ing)"
# A word that turns what follows around ("I won't vote Bob", "Carol, not Bob", "It's not hail").
_NEG_CUE = r"(?:\bnot|\bnever|n't|\bcannot|\brather\s+than|\binstead\s+of)\b"
# A negation reaches to the end of its clause.
_CLAUSE_BREAK = r"[.;:!?,\n]|\bbut\b"
# What may stand between a negation and the name it negates ("not Bob", "not for **Bob**").
_NAME_LEAD = r"""(?:\s+(?:for|on|against|with))?[\s*_'"`@(\[-]*$"""


def _name_re(name: str) -> str:
    return (
        r"(?<![A-Za-z])" + r"\s+".join(map(re.escape, name.split())) + r"(?![A-Za-z])"
    )


def _clause_before(text: str, pos: int) -> str:
    return re.split(_CLAUSE_BREAK, text[:pos], flags=re.IGNORECASE)[-1]


def _negated_command(text: str, pos: int, verb: str, names: List[str]) -> bool:
    """Whether the command word at ``pos`` is negated: a negation at most three words before it in its clause, with
    no command word or player name in between ("I won't vote", "I do not want to vote", "don't kill")."""
    m = re.search(
        rf"{_NEG_CUE}((?:\W+\w+){{0,3}})\W*$", _clause_before(text, pos), re.IGNORECASE
    )
    if not m:
        return False
    between = m.group(1)
    return not re.search(rf"\b(?:{verb})\b", between, re.IGNORECASE) and not any(
        re.search(_name_re(n), between) for n in names
    )


def read_named_player(
    text: str, names: List[str], verb: str = _VOTE_VERB
) -> Tuple[str | None, List[str]]:
    """The player a move names, read as a person at the table would read it.

    The name right after the command word counts ("vote Bob", "vote for Bob", "vote: **'Bob'**"), wherever it
    stands in the reply, unless the command is negated: "I won't vote Bob, vote Carol" is a vote for Carol.
    Otherwise the reply counts for a player only if it is just that name or names exactly one player (written with
    a capital, so the word "mark" is not a vote for Mark), and a negated mention does not count: "My vote goes to
    Carol, not Bob" names Carol. Names match as whole words, so punctuation, quotes, markdown and a trailing "'s" do
    not matter. Returns (player, []) or, when no single player is named, (None, the players mentioned).
    """
    text = (text or "").translate(_QUOTES)
    hits: List[Tuple[int, str]] = []
    negated: List[Tuple[int, int]] = []
    for n in names:
        for m in re.finditer(
            rf"\b(?:{verb})\b{_GAP}{_name_re(n)}", text, re.IGNORECASE
        ):
            if not _negated_command(text, m.start(), verb, names):
                hits.append((m.start(), n))
                continue
            stop = re.search(_CLAUSE_BREAK, text[m.end() :], re.IGNORECASE)
            negated.append((m.start(), m.end() + stop.start() if stop else len(text)))
    if hits:
        return min(hits)[1], []
    words = re.sub(r"[\W_]+", " ", text).strip().lower()
    for n in names:
        if words == " ".join(n.lower().split()):
            return n, []
    mentioned = [
        n
        for n in names
        if any(
            not re.search(
                _NEG_CUE + _NAME_LEAD, _clause_before(text, m.start()), re.IGNORECASE
            )
            and not any(a <= m.start() < b for a, b in negated)
            for m in re.finditer(_name_re(n), text)
        )
    ]
    if len(mentioned) == 1:
        return mentioned[0], []
    return None, mentioned


def vote_problem(text: str, mentioned: List[str], verb: str = _VOTE_VERB) -> str:
    """Why a vote names no single player, for the public record."""
    if mentioned:
        return (
            f"names several players ({', '.join(mentioned)}) and no single 'vote NAME'"
        )
    if re.search(rf"\b(?:{verb})\b", text or "", re.IGNORECASE):
        return "names no player"
    return "is not a vote"


def clean_clue(text: str) -> str:
    """The clue a player said: the reply without a command word ("clue sleet", "Clue: drizzle", "My clue is fog"),
    quotes, markdown or end punctuation. Earlier only the first word was kept, so "clue sleet" became "clue"."""
    t = (text or "").translate(_QUOTES).strip()
    t = (
        re.sub(
            r"^\s*(?:(?:my|the)\s+)?(?:clue|word)\b(?:\s+is\b)?\s*[:\-]?\s*",
            "",
            t,
            flags=re.IGNORECASE,
        )
        or t
    )
    t = re.sub(r"^\s*say\b\s*[:\-]?\s*", "", t, flags=re.IGNORECASE) or t
    return re.sub(r"\s+", " ", t.strip(" \t\n\"'*`_.!?,;:()[]")).strip()


_GUESS_LEAD = (
    r"^\s*(?:i\s+(?:think|believe|guess|bet)|i'?m\s+guessing|i\s+am\s+guessing|my\s+(?:final\s+)?(?:guess|answer)"
    r"(?:\s+is)?|final\s+answer|guess(?:ing)?|answer|is\s+it)\b\s*(?:that\b)?\s*[:\-]?\s*"
)
_WORD_IS = (
    r"^\s*(?:(?:the\s+(?:secret\s+)?(?:word|answer)|it)\s+(?:is|might\s+be|could\s+be|may\s+be|must\s+be|would\s+be|'s)"
    r"|it's)\b\s*[:\-]?\s*"
)


def drop_negated(text: str) -> str:
    """The text without its negated parts: each negation and the rest of its clause, so "It's not hail, it's sleet"
    becomes "It's , it's sleet" and "I don't know" becomes "I"."""
    return re.sub(
        rf"\w*{_NEG_CUE}.*?(?={_CLAUSE_BREAK}|$)", " ", text or "", flags=re.IGNORECASE
    )


def guess_phrase(text: str) -> str:
    """What a guess names: the first clause of the reply that says something once lead-ins ("guess", "I think the
    word is", "It's", "My guess:"), quotes, markdown and an article are removed. A negated part does not count.
    "I guess the word is sleet." and "guess 'sleet'" give "sleet"; "sleet, because Bob said ice" gives "sleet";
    "It's not hail, it's sleet" gives "sleet"."""
    t = re.sub(r"[*_`]", " ", drop_negated((text or "").translate(_QUOTES)).lower())
    for clause in re.split(
        r"[.,;!?()\n]|\s[-–—]\s|\b(?:because|since|as|given|based)\b", t
    ):
        previous = None
        while previous != clause:
            previous = clause
            clause = re.sub(_WORD_IS, "", re.sub(_GUESS_LEAD, "", clause))
        clause = re.sub(r"^(?:a|an|the)\s+", "", clause.strip(" \t\"'"))
        clause = re.sub(r"\s+", " ", clause.strip(" \t\"'"))
        if clause:
            return clause
    return ""


def _word_keys(word: str) -> set[str]:
    """A card word and its close variants (case, plural, hyphen or space), spaces removed."""
    w = re.sub(r"[\s\-]+", "", word.lower())
    keys = {w, w + "s", w + "es"}
    if w.endswith("y"):
        keys.add(w[:-1] + "ies")
    return keys


def read_card_guess(text: str, card_words: List[str]) -> str | None:
    """The card word a caught Chameleon's reply names, or None when it names no single card word.

    The guess phrase must be a card word or a close variant of exactly one ("sleets", "hash-brown"); failing that,
    a reply that mentions exactly one card word outside its negated parts names that word ("Not hail. My guess is
    sleet." names sleet). Earlier the whole reply had to equal the word once "guess" was deleted, so "guess sleet."
    and "Guess: sleet" lost.
    """
    phrase = re.sub(r"[\s\-]+", "", guess_phrase(text))
    exact = [w for w in card_words if re.sub(r"[\s\-]+", "", w.lower()) == phrase]
    close = [w for w in card_words if phrase in _word_keys(w)]
    if exact or len(close) == 1:
        return (exact or close)[0].lower()
    if close:
        return None
    low = drop_negated((text or "").translate(_QUOTES)).lower()
    mentioned = [
        w
        for w in card_words
        if re.search(
            r"(?<![a-z])"
            + r"[\s\-]*".join(map(re.escape, w.lower().split()))
            + r"(?:e?s)?(?![a-z])",
            low,
        )
    ]
    return mentioned[0].lower() if len(mentioned) == 1 else None


# ============================================================================
# Action Handler
# ============================================================================


def _acting(env: SocialDeductionGame, agent_name: str) -> bool:
    """Whether the engine asked this player to act in the step being processed."""
    try:
        return bool(env.action_mask[env.agents.index(agent_name)])
    except (ValueError, IndexError):
        return False


class ChameleonActionHandler(ActionHandler):
    def handle_action(
        self, env: SocialDeductionGame, agent_name: str, action: AgentAction
    ) -> None:
        if not isinstance(env, ChameleonEnv) or not _acting(env, agent_name):
            return
        # A failed model call arrives as 'none'; it is read as an empty reply, and every phase says what that means.
        text = action.argument if action.action_type in ["action", "speak"] else ""

        if env.current_state == "Clue":
            clue = clean_clue(text)
            env.internal_state.setdefault("clues", {})[agent_name] = clue
            if not clue:
                env.recv_message(
                    "Environment",
                    SimpleMessage(message=f"[Game] {agent_name} gave no clue."),
                )
            elif len(clue.split()) > 1:
                # Shown, not blocked: saying more than one word is the player's own mistake.
                env.recv_message(
                    "Environment",
                    SimpleMessage(
                        message=f'[Game] {agent_name}\'s clue "{clue}" is more than one word. '
                        f"It stands as given."
                    ),
                )

        elif env.current_state in ("Vote", "TieBreak"):
            if env.current_state == "TieBreak" and agent_name != env.internal_state.get(
                "dealer_name"
            ):
                return
            target, mentioned = read_named_player(text, list(env.agents))
            if env.current_state == "TieBreak":
                env.internal_state["tie_break_reply"] = text
                if target in env.internal_state.get("tied_candidates", []):
                    env.internal_state["tie_break_choice"] = target
            elif target:
                env.internal_state.setdefault("votes", {})[agent_name] = target
            else:
                env.internal_state.setdefault("vote_problems", {})[agent_name] = (
                    vote_problem(text, mentioned)
                    if text.strip()
                    else "was empty or missing"
                )

        elif env.current_state == "Guess":
            chameleon_name = env.internal_state.get("chameleon_name")
            if agent_name != chameleon_name:
                return
            words = [
                w.strip()
                for w in env._config.get("word_list", "").split(",")
                if w.strip()
            ]
            env.internal_state["chameleon_guess"] = read_card_guess(text, words)
            env.internal_state["chameleon_guess_reply"] = text

    def get_action_instruction(self, env: SocialDeductionGame, agent_name: str) -> str:
        if not isinstance(env, ChameleonEnv):
            return ""

        if env.current_state == "Clue":
            clues = env.internal_state.get("clues", {})
            given = [f"{n}: {c or '(no clue)'}" for n, c in clues.items()]
            clue_str = f" Clues so far: {', '.join(given)}." if given else ""
            return (
                f"Give a ONE-WORD clue related to the secret word.{clue_str} "
                f"You may repeat another player's clue. "
                f"If you are a Citizen, prove you know the word without making it too obvious. "
                f"If you are the Chameleon, give a plausible clue to blend in."
            )
        elif env.current_state == "Discussion":
            clues = env.internal_state.get("clues", {})
            clue_str = ", ".join(f"{n}: {c or '(no clue)'}" for n, c in clues.items())
            return (
                f"All clues: [{clue_str}]. "
                f"Discuss who you think the Chameleon is based on these clues."
            )
        elif env.current_state == "Vote":
            player_list = ", ".join(list(env.agents))
            return (
                f"Vote for who you think is the Chameleon. Players: [{player_list}]. "
                f"Use 'vote NAME' (e.g., 'vote Alice'). A vote that names no single player counts as no vote."
            )
        elif env.current_state == "TieBreak":
            dealer = env.internal_state.get("dealer_name")
            if agent_name == dealer:
                tied = ", ".join(env.internal_state.get("tied_candidates", []))
                return (
                    f"As dealer, decide the tied vote. Choose one of [{tied}]. "
                    f"Use 'vote NAME'. If your reply names none of them, one of them is drawn at random."
                )
            return f"Waiting for dealer {dealer} to decide the tied vote."
        elif env.current_state == "Guess":
            chameleon_name = env.internal_state.get("chameleon_name")
            if agent_name == chameleon_name:
                clues = env.internal_state.get("clues", {})
                clue_str = ", ".join(
                    f"{n}: {c or '(no clue)'}" for n, c in clues.items()
                )
                word_list = env._config.get("word_list", "")
                return (
                    f"You were identified as the Chameleon! You get one last chance. "
                    f"All clues were: [{clue_str}]. "
                    f"Possible words: [{word_list}]. "
                    f"Use 'guess WORD' to guess the secret word. A reply that names no single word from the "
                    f"card counts as a wrong guess."
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
        self.agent_message_buffer.clear()
        self._round_robin_idx = 0
        super().reset(
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
            "vote_problems": {},
            "chameleon_name": chameleon_name,
            "chameleon_guess": None,
            "dealer_name": self.agents[0],
            "tied_candidates": [],
            "tie_break_choice": None,
            "game_over": False,
            "final_scores": {},
            "end_reason": "",
        }

        # The topic card is public; only the identity of its secret word is private.
        category = self._config.get("category", "")
        secret_word = self._config.get("secret_word", "")
        word_list = self._config.get("word_list", "")
        self.recv_message(
            "Environment",
            SimpleMessage(
                message=f"[Topic Card] Category: {category}. Words: [{word_list}]. "
                f"Dealer: {self.internal_state['dealer_name']}. "
                f"Clue order: {', '.join(self.agents)}. "
                f"The dealer gives the first clue and decides any tied vote."
            ),
        )

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

        return self._build_observations()

    def _update_action_mask(self) -> None:
        """Only the dealer breaks ties; only the Chameleon guesses."""
        super()._update_action_mask()
        if self.current_state in ("Guess", "TieBreak"):
            acting_name = self.internal_state.get(
                "chameleon_name" if self.current_state == "Guess" else "dealer_name"
            )
            agents = list(self.agents)
            for idx, name in enumerate(agents):
                self.action_mask[idx] = name == acting_name

    def _perform_transition_state(self) -> None:
        if self.current_state == "Vote" and self.internal_state.get("tied_candidates"):
            self.current_state = "TieBreak"
            self._state_turn_count[self.current_state] = 0
            self._round_robin_idx = 0
            self.recv_message(
                "Environment", SimpleMessage(message="[Game] Entering state: TieBreak")
            )
        else:
            super()._perform_transition_state()

    def _check_eliminations(self) -> None:
        if not self._should_transition_state():
            return

        if self.current_state == "Vote":
            votes = self.internal_state.get("votes", {})
            # No valid votes means everyone is tied at zero; the dealer still
            # chooses an accused player before any escape guess can happen.
            vote_counts = Counter({name: 0 for name in self.agents})
            vote_counts.update(votes.values())
            vote_count = max(vote_counts.values())
            tied = sorted(n for n, c in vote_counts.items() if c == vote_count)
            # Every voter is listed, with the player the vote was read as or why it counts as no vote.
            problems = self.internal_state.get("vote_problems", {})
            vote_summary = ", ".join(
                f"{voter}: {votes[voter]}"
                if voter in votes
                else f"{voter}: no valid vote (the reply {problems.get(voter, 'was empty or missing')})"
                for voter in self.agents
            )
            if not votes:
                vote_summary += ". No valid votes"
            self.recv_message(
                "Environment",
                SimpleMessage(message=f"[Vote Results] {vote_summary}."),
            )
            if len(tied) > 1:
                self.internal_state["tied_candidates"] = tied
                self.recv_message(
                    "Environment",
                    SimpleMessage(
                        message=f"[Vote Tie] {', '.join(tied)} each received "
                        f"{vote_count} votes. Dealer "
                        f"{self.internal_state['dealer_name']} must choose among them."
                    ),
                )
            else:
                self._resolve_vote(tied[0])

        elif self.current_state == "TieBreak":
            tied = self.internal_state["tied_candidates"]
            choice = self.internal_state.get("tie_break_choice")
            if choice not in tied:
                # Benchmark default for an unparseable/absent dealer action.
                # Keep the fallback explicit and reproducible, without a retry.
                rng = random.Random(f"{self._config.get('seed')}-vote_tie")
                choice = rng.choice(tied)
                self.internal_state["tie_break_fallback"] = True
                self.recv_message(
                    "Environment",
                    SimpleMessage(
                        message=f"[Vote Tie] Dealer submitted no valid tied candidate. "
                        f"Seeded random fallback selected {choice}."
                    ),
                )
            else:
                self.recv_message(
                    "Environment",
                    SimpleMessage(
                        message=f"[Vote Tie] Dealer {self.internal_state['dealer_name']} "
                        f"selected {choice}."
                    ),
                )
            self._resolve_vote(choice)

        elif self.current_state == "Guess":
            guess = self.internal_state.get("chameleon_guess")
            secret_word = self._config.get("secret_word", "").lower()
            chameleon_name = self.internal_state.get("chameleon_name")
            scores: Dict[str, float] = {}

            # Default-and-continue: a reply that names no single card word (or no reply) is a wrong guess, said so
            # in the result line, so the game ends instead of stalling in the Guess state until the turn cap.
            if guess is None:
                reply = (self.internal_state.get("chameleon_guess_reply") or "").strip()
                guess = (
                    f'(no single word from the card in the reply "{reply[:80]}")'
                    if reply
                    else "(no reply)"
                )

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

    def _resolve_vote(self, accused: str) -> None:
        chameleon_name = self.internal_state.get("chameleon_name")
        if accused == chameleon_name:
            self.recv_message(
                "Environment",
                SimpleMessage(
                    message=f"[Game] The group correctly identified {chameleon_name} "
                    f"as the Chameleon! But the Chameleon gets one last chance "
                    f"to guess the secret word..."
                ),
            )
            return

        chameleon_team = [
            n for n in self.agents if self.agent_to_role.get(n, "") == "Chameleon"
        ]
        citizen_team = [
            n for n in self.agents if self.agent_to_role.get(n, "") != "Chameleon"
        ]
        scores = {name: 1.0 / len(chameleon_team) for name in chameleon_team}
        scores.update({name: -1.0 / len(citizen_team) for name in citizen_team})
        reason = (
            f"The group voted for {accused}, but the Chameleon was "
            f"{chameleon_name}! Chameleon wins!"
        )
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
