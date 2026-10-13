"""Launcher for the Insider social game scenario."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
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
        env = kwargs.get("env")
        # A game decided on the cap turn keeps its result.
        if env and env.internal_state.get("game_over", False):
            agent_names = list(env.agents)
            scores = env.internal_state.get("final_scores", {})
            reason = env.internal_state.get("end_reason", "Game over.")
            response: List[
                Tuple[str, Tuple[Tuple[str, int | float | bool], str]]
            ] = [("environment", (("terminated", True), reason))]
            for idx, name in enumerate(agent_names):
                key = f"agent_{idx + 1}"
                score = scores.get(name, -1.0)
                response.append((key, (("complete_rating", score), "")))
            return response

        if turn_number >= self.max_turn_number:
            if env:
                # Timeout = everyone loses (the official sand-timer rule). Every phase now
                # ends in a fixed number of turns, so this is a safety net only.
                reason = (
                    "Max turns reached before the vote finished."
                    if env.internal_state.get("word_guessed", False)
                    else "Max turns reached. Word not guessed."
                )
                response = [("environment", (("terminated", True), reason))]
                for idx, name in enumerate(env.agents):
                    response.append(
                        (f"agent_{idx + 1}", (("complete_rating", -1.0), "Timeout"))
                    )
                return response
            return [("environment", (("terminated", True), "Max turns reached."))]

        return [("environment", (("terminated", False), ""))]


# ============================================================================
# Secret word and the Master's answers
# ============================================================================


def draw_secret_word(
    config: Dict[str, Any], options: Dict[str, str] | None = None, seed: int | None = None
) -> Tuple[str, str]:
    """(word, category) for this episode.

    Precedence for repeatability:
      1. options/config explicit override: "secret_word" (+ optional "secret_word_category")
         lets a roster pin a specific word.
      2. else, sample deterministically from "secret_word_pool" using the seed (config["seed"],
         from the roster, when none is given). The pool is sorted by word before sampling so
         the seed -> word mapping is stable when new words are appended to the pool.
    """
    explicit = (options or {}).get("secret_word", "") or config.get("secret_word", "")
    pool = config.get("secret_word_pool", [])
    if explicit:
        cat_lookup = {e.get("word", ""): e.get("category", "") for e in pool}
        return explicit, cat_lookup.get(explicit, config.get("secret_word_category", ""))
    if pool:
        effective_seed = seed if seed is not None else config.get("seed")
        entry = random.Random(effective_seed).choice(
            sorted(pool, key=lambda e: e.get("word", ""))
        )
        return entry.get("word", ""), entry.get("category", "")
    return "", config.get("secret_word_category", "")


def pin_secret_word(config: Dict[str, Any]) -> None:
    """Draw the word before the prompts are built, so the Insider's prompt names it.

    The draw used to happen in reset(), after create_agents() had filled the Insider's
    secret line from config["secret_word"], which no roster sets ("The secret word is: .").
    """
    word, category = draw_secret_word(config)
    if word:
        config["secret_word"] = word
        config["secret_word_category"] = category


_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})
_SUBJECT_NOUNS = (
    "word answer thing item object animal creature insect bird mammal fish food dish meal "
    "fruit vehicle place location building landmark activity hobby pastime sport game "
    "phenomenon instrument profession job occupation career"
).split()
_SUBJECT_ADJECTIVES = "secret exact mystery hidden target correct actual".split()
_ADVERBS = "exactly specifically simply just actually really commonly usually literally precisely".split()


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.translate(_QUOTES).lower().replace("-", " ")).strip()


def _word_keys(word: str) -> set[str]:
    """The secret word and its close variants (case, plural, hyphen or space), spaces removed."""
    w = _norm(word).replace(" ", "")
    keys = {w, w + "s", w + "es"}
    if w.endswith("y"):
        keys.add(w[:-1] + "ies")
    return keys


def names_secret_word(question: str, word: str, category: str = "") -> bool:
    """Whether ``question`` asks directly whether the answer is the secret word.

    Matches "Is it (a) X?", "Is the secret word 'X'?", "Is the <category> (called) X?",
    "Could it be X?" and "Is X the secret word?", where X is the word or a close variant
    and nothing else follows. Questions that only mention the word ("Is it larger than
    a violin?", "Is it a type of pizza?") are left to the judge.
    """
    if not word:
        return False
    q = _norm(question)
    if q.count("?") > 1:
        return False
    q = re.split(r"(?<=[.!:])\s+", q)[-1]  # the question itself, after any preamble
    cat = _norm(category).split()
    nouns = "|".join(map(re.escape, _SUBJECT_NOUNS + cat))
    adjectives = "|".join(map(re.escape, _SUBJECT_ADJECTIVES + cat))
    subject = (
        rf"(?:it|this|that|(?:the|this|our|your)\s+(?:(?:{adjectives})\s+)*(?:{nouns})"
        rf"(?:\s+(?:word|answer))?)\b"
    )
    lead = r"^(?:(?:so|then|ok|okay|alright|well),?\s+)?"
    end = r"""['"]?[\s?.!'"})\]]*$"""
    direct = (
        rf"{lead}(?:is\s+{subject}|(?:could|would|might|can)\s+{subject}\s+be)"
        rf"(?:\s+(?:{'|'.join(_ADVERBS)}))?(?:\s+(?:called|named|known as))?"
        rf"(?:\s+(?:a|an|the)\b)?\s+['\"]?\s*(?P<cand>[a-z0-9][a-z0-9 ]*?)\s*{end}"
    )
    reverse = (
        rf"{lead}is\s+(?:(?:a|an|the)\s+)?['\"]?(?P<cand>[a-z0-9][a-z0-9 ]*?)['\"]?\s+"
        rf"(?:the|our)\s+(?:secret\s+)?(?:word|answer){end}"
    )
    m = re.match(direct, q) or re.match(reverse, q)
    if not m:
        return False
    cand = re.sub(r"^(?:a|an|the)\s+", "", m.group("cand").strip())
    return cand.replace(" ", "") in _word_keys(word)


_GUESS_LEAD = (
    r"^\s*(?:i\s+(?:think|believe|guess|bet)|i'?m\s+guessing|i\s+am\s+guessing|my\s+(?:final\s+)?(?:guess|answer)"
    r"(?:\s+is)?|final\s+answer|guess(?:ing)?|answer|is\s+it)\b\s*(?:that\b)?\s*[:\-]?\s*"
)
_WORD_IS = (
    r"^\s*(?:(?:the\s+(?:secret\s+)?(?:word|answer)|it)\s+(?:is|might\s+be|could\s+be|may\s+be|must\s+be|would\s+be|'s)"
    r"|it's)\b\s*[:\-]?\s*"
)


def guess_phrase(text: str) -> str:
    """What a guess names: the first clause of the reply that says something once lead-ins ("guess", "I think the
    word is", "It's", "My guess:"), quotes, markdown and an article are removed. A negated part does not count.
    "I think the word might be 'elephant'. It's a mammal" gives "elephant"; "guess Aurora Borealis" gives "aurora
    borealis"; "It's not a koala, it's a kangaroo" gives "kangaroo"."""
    t = re.sub(r"[*_`]", " ", drop_negated((text or "").translate(_QUOTES)).lower())
    for clause in re.split(r"[.,;!?()\n]|\s[-–—]\s|\b(?:because|since|as|given|based)\b", t):
        previous = None
        while previous != clause:
            previous = clause
            clause = re.sub(_WORD_IS, "", re.sub(_GUESS_LEAD, "", clause))
        clause = re.sub(r"^(?:a|an|the)\s+", "", clause.strip(" \t\"'"))
        clause = re.sub(r"\s+", " ", clause.strip(" \t\"'"))
        if clause:
            return clause
    return ""


def guess_names_word(text: str, word: str) -> bool:
    """Whether a guess names the secret word, with the variants the question rule accepts (case, plural, hyphen or
    space). The guess used to have to equal the word once "guess" was deleted, so "guess 'castle'", "guess
    pancakes" and "guess sailboat." were scored wrong."""
    return bool(word) and _norm(guess_phrase(text)).replace(" ", "") in _word_keys(word)


# ----------------------------------------------------------------------------
# Votes
# ----------------------------------------------------------------------------

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


def drop_negated(text: str) -> str:
    """The text without its negated parts: each negation and the rest of its clause, so "It's not hail, it's sleet"
    becomes "It's , it's sleet" and "I don't know" becomes "I"."""
    return re.sub(
        rf"\w*{_NEG_CUE}.*?(?={_CLAUSE_BREAK}|$)", " ", text or "", flags=re.IGNORECASE
    )


# ----------------------------------------------------------------------------
# The stand-in Master (LLM judge)
# ----------------------------------------------------------------------------

JUDGE_DEFAULT_MODEL = "gpt-6-luna"


def judge_request(config: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """The judge model and the extra request fields sent with every question.

    Default: config["judge_model"] (gpt-6-luna, a reasoning model, through the OpenAI API) with
    config["judge_params"], which is empty: no temperature and no token limit. Environment overrides:

    - SOTOPIA_INSIDER_JUDGE: the model, in litellm form (e.g. openai/Qwen3.8-27B for a vLLM server);
    - SOTOPIA_INSIDER_JUDGE_BASE_URL: an OpenAI-compatible base URL, with SOTOPIA_INSIDER_JUDGE_API_KEY
      (default "EMPTY");
    - SOTOPIA_INSIDER_JUDGE_PARAMS: a JSON object merged over judge_params (a null value removes a field). For
      Qwen3.8-27B with thinking on:
      '{"max_tokens": 4096, "temperature": 0, "extra_body": {"chat_template_kwargs": {"enable_thinking": true}}}'

    Only the final answer is read (see parse_judge_reply), never the reasoning.
    """
    model = os.environ.get("SOTOPIA_INSIDER_JUDGE") or config.get("judge_model") or JUDGE_DEFAULT_MODEL
    params: Dict[str, Any] = dict(config.get("judge_params") or {})
    override = os.environ.get("SOTOPIA_INSIDER_JUDGE_PARAMS", "").strip()
    if override:
        extra = json.loads(override)
        if not isinstance(extra, dict):
            raise ValueError("SOTOPIA_INSIDER_JUDGE_PARAMS must be a JSON object")
        for key, value in extra.items():
            if value is None:
                params.pop(key, None)
            else:
                params[key] = value
    if os.environ.get("SOTOPIA_INSIDER_JUDGE_BASE_URL"):
        params["api_base"] = os.environ["SOTOPIA_INSIDER_JUDGE_BASE_URL"]
        params["api_key"] = os.environ.get("SOTOPIA_INSIDER_JUDGE_API_KEY", "EMPTY")
    return model, params


JUDGE_PROMPT = (
    "You are the Master in a game of Insider, a word-guessing game like 20 Questions. "
    'The secret word is "{word}"{category}. The other players do not know it; they ask '
    "yes/no questions to find it. Answer each question truthfully about the thing the secret "
    "word names, as the Master would. If the player asks whether the answer is the secret "
    'word itself (for example "Is it a {word}?" or "Is the secret word {word}?"), the '
    "answer is Yes.\n"
    "Reply with exactly one of: Yes, No, I don't know. Use I don't know only when the "
    "question cannot be answered with yes or no, or nobody could know the answer.\n\n"
    "{history}"
    "Question from {asker}: {question}\n"
    "Answer:"
)


_DONT_KNOW = (
    r"i\s+don't\s+know|i\s+do\s+not\s+know|don't\s+know|do\s+not\s+know|no\s+idea|not\s+sure|unsure|unclear|unknown"
    r"|uncertain|can't\s+(?:say|tell)|cannot\s+(?:say|tell|be\s+determined)"
)


def parse_judge_reply(reply: str | None) -> str | None:
    """The judge's final answer as one of the Master's three answers, or None when it is none of them.

    Reasoning is never read: a <think> block (a local model served without a reasoning parser) is dropped, and
    so is a leading "Answer:" or "The answer is". "No idea" is I don't know (it used to read as No), "Correct" and
    "Y" are Yes, and an empty or unmatched reply is None, which the caller retries and then marks as unreadable
    (it used to be shown as an ordinary "I don't know").
    """
    text = (reply or "").translate(_QUOTES)
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    if "</think>" in text.lower():
        text = re.split(r"</think>", text, flags=re.IGNORECASE)[-1]
    text = re.sub(r"[*_`#>]", " ", text).strip().lower()
    text = re.sub(
        r"^[\W_]*(?:(?:final\s+)?answer\s*[:\-]|(?:the|my)\s+(?:final\s+)?answer\s+is\b\s*:?|master\s*:)\s*", "", text
    )
    text = re.sub(r"^[\W_]+", "", text)
    if re.match(rf"(?:{_DONT_KNOW})\b", text):
        return "I don't know"
    if re.match(r"(?:yes|yeah|yep|correct|true)\b", text) or text in ("y", "y."):
        return "Yes"
    if re.match(r"(?:no|nope|incorrect|false)\b", text) or text in ("n", "n."):
        return "No"
    return None


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

        if env.current_state == "Questioning":
            # The player whose turn it is asks; every turn counts toward the question
            # budget, so an empty or failed turn cannot stretch the phase.
            if not self._is_acting(env, agent_name):
                return
            question = (
                action.argument.strip()
                if action.action_type in ["action", "speak"]
                else ""
            )
            env.internal_state["questions_asked"] = (
                env.internal_state.get("questions_asked", 0) + 1
            )
            max_q = env._config.get("max_questions", 15)
            asked = env.internal_state["questions_asked"]
            if question:
                answer, source = self._answer_question(env, agent_name, question)
                env.internal_state.setdefault("qa_log", []).append(
                    {"asker": agent_name, "question": question, "answer": answer, "source": source}
                )
                if source == "judge_error":
                    message = f"[Answer] {answer} (no answer could be produced: the judge call failed)"
                elif source == "judge_unreadable":
                    message = (
                        f"[Answer] {answer} (no answer could be produced: the judge's reply was not Yes, No "
                        f"or I don't know)"
                    )
                else:
                    message = f"[Answer] {answer}"
                env.recv_message("Environment", SimpleMessage(message=message))
            else:
                env.recv_message(
                    "Environment",
                    SimpleMessage(
                        message=f"[Game] {agent_name} asked no question. The turn still counts "
                        f"as question {asked} of {max_q}."
                    ),
                )

            # Check if question limit reached
            if asked >= max_q:
                env.recv_message(
                    "Environment",
                    SimpleMessage(
                        message=f"[Game] Question limit ({max_q}) reached! "
                        f"Time to guess the word."
                    ),
                )
            return

        if not self._is_acting(env, agent_name):
            return
        # A failed model call arrives as 'none'; it is read as an empty reply and announced below.
        text = action.argument.strip() if action.action_type in ["action", "speak"] else ""

        if env.current_state == "Guess":
            arg = text.lower()
            secret_word = env.internal_state.get("secret_word", "")

            if not arg:
                env.recv_message(
                    "Environment",
                    SimpleMessage(message=f"[Game] {agent_name} sent no guess. Counted as a pass."),
                )
                return
            if arg.startswith("pass") or arg.startswith("skip"):
                return

            if guess_names_word(text, secret_word):
                env.internal_state["word_guessed"] = True
                env.internal_state["guesser"] = agent_name
                env.recv_message(
                    "Environment",
                    SimpleMessage(
                        message=f"[Game] {agent_name} correctly guessed the word: {secret_word.lower()}! "
                        f"Now discuss who you think the Insider is."
                    ),
                )
            else:
                if "incorrect_guesses" not in env.internal_state:
                    env.internal_state["incorrect_guesses"] = []
                env.internal_state["incorrect_guesses"].append((agent_name, guess_phrase(text)))

        elif env.current_state == "Vote":
            target, mentioned = read_named_player(text, list(env.agents))
            if target:
                env.internal_state.setdefault("votes", {})[agent_name] = target
            else:
                # The reason is announced when the vote resolves (an abstention).
                if not text:
                    reason = "sent no vote"
                elif mentioned:
                    reason = f"named several players ({', '.join(mentioned)}) without a single 'vote NAME'"
                else:
                    reason = "named no player"
                env.internal_state.setdefault("vote_problems", {})[agent_name] = reason

    @staticmethod
    def _is_acting(env: SocialDeductionGame, agent_name: str) -> bool:
        """Whether it is this agent's turn (the mask is still this turn's)."""
        try:
            return bool(env.action_mask[list(env.agents).index(agent_name)])
        except (ValueError, IndexError):
            return False

    def _answer_question(
        self, env: InsiderEnv, asker: str, question: str
    ) -> Tuple[str, str]:
        """The Master's answer to a yes/no question: (answer, source).

        A question that asks directly whether the answer is the secret word (or a close
        variant) is answered Yes in code (source "rule"). Every other question goes to an
        LLM judge (default gpt-6-luna, see judge_request) told the game, the word, its category and
        this episode's earlier questions and answers (source "judge"). Each failed call or unreadable
        reply is retried once; then the answer is "I don't know", marked as a failed call
        (source "judge_error") or as an unreadable reply (source "judge_unreadable").
        """
        secret_word = env.internal_state.get(
            "secret_word", env._config.get("secret_word", "")
        )
        category = env.internal_state.get("secret_word_category", "")
        if not secret_word:
            return "I don't know", "judge_error"
        if names_secret_word(question, secret_word, category):
            return "Yes", "rule"

        judge_model, judge_params = judge_request(env._config)
        earlier = env.internal_state.get("qa_log", [])
        history = (
            "Earlier questions and the answers given:\n"
            + "".join(f"- {e['asker']}: {e['question']} -> {e['answer']}\n" for e in earlier)
            + "\n"
            if earlier
            else ""
        )
        prompt = JUDGE_PROMPT.format(
            word=secret_word,
            category=f" (category: {category})" if category else "",
            history=history,
            asker=asker,
            question=question.strip(),
        )

        source = "judge_error"
        for attempt in range(2):
            try:
                resp = litellm_completion(
                    model=judge_model,
                    messages=[{"role": "user", "content": prompt}],
                    **judge_params,
                )
            except Exception as exc:
                logging.getLogger(__name__).warning(
                    "Insider judge call failed (attempt %d: %s).", attempt + 1, exc
                )
                continue
            reply = resp.choices[0].message.content or ""
            answer = parse_judge_reply(reply)
            if answer is not None:
                return answer, "judge"
            source = "judge_unreadable"
            logging.getLogger(__name__).warning(
                "Insider judge reply not readable (attempt %d): %r", attempt + 1, reply[:200]
            )
        return "I don't know", source

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
                f"The environment will answer truthfully: Yes, No, or I don't know. "
                f"Your turn counts as one of the {max_q} questions even if you ask nothing."
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
                f"Use 'vote NAME' (e.g., 'vote Alice'). A vote that names no single player "
                f"counts as an abstention. The player with the most votes is accused; if the top "
                f"vote is tied, one of the tied players is drawn at random (announced to everyone)."
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

        # Draw the secret word for this episode. prepare_scenario() pins it into the config
        # before the prompts are built, so this normally returns that word. The tournament
        # runner calls reset() with seed=None, which falls back to config["seed"].
        secret_word, category = draw_secret_word(self._config, options, seed)

        self.internal_state = {
            "questions_asked": 0,
            "word_guessed": False,
            "guesser": None,
            "incorrect_guesses": [],
            "votes": {},
            "vote_problems": {},
            "game_over": False,
            "final_scores": {},
            "end_reason": "",
            "secret_word": secret_word,
            "secret_word_category": category,
            "qa_log": [],
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

        # The base reset built each agent's action instruction before the word was drawn,
        # so the turn-0 questioner was never told the category; rebuild it.
        if self.action_handler:
            for agent_name in self.agents:
                if agent_name in obs:
                    obs[agent_name].action_instruction = (
                        self.action_handler.get_action_instruction(self, agent_name)
                    )

        return obs

    def _check_eliminations(self) -> None:
        """Handle state transitions based on game progress."""
        # Question limit reached -> Guess, as soon as it is reached (not only at the end
        # of a full cycle of turns, which let uncounted turns start a fourth cycle).
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
            return

        if not self._should_transition_state():
            return

        if self.current_state == "Guess":
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
            # A vote that names no single player (or no reply at all) is an abstention, announced
            # with its reason, and the vote resolves on this turn. (It used to repeat the Vote phase
            # with no feedback until the turn cap, which scored the game "Word not guessed".)
            problems = self.internal_state.get("vote_problems", {})
            by_reason: Dict[str, List[str]] = {}
            for n in self.agents:
                if n not in votes:
                    by_reason.setdefault(problems.get(n, "sent no vote"), []).append(n)
            for reason, abstained in by_reason.items():
                self.recv_message(
                    "Environment",
                    SimpleMessage(
                        message=f"[Game] {', '.join(abstained)} {reason} and "
                        f"{'abstains' if len(abstained) == 1 else 'abstain'}."
                    ),
                )
            if votes:
                # How each vote was read, so a misread is visible.
                self.recv_message(
                    "Environment",
                    SimpleMessage(
                        message="[Game] Votes counted: "
                        + ", ".join(f"{v} -> {votes[v]}" for v in self.agents if v in votes)
                        + "."
                    ),
                )
                vote_counts = Counter(votes.values())
                vote_count = max(vote_counts.values())
                tied = [n for n, c in vote_counts.items() if c == vote_count]
                most_voted: str | None = tied[0]
                if len(tied) > 1:
                    most_voted = self.break_tie(tied)
                    self.recv_message(
                        "Environment",
                        SimpleMessage(
                            message=f"[Game] Vote tied between {', '.join(tied)} "
                            f"({vote_count} each). Random draw: {most_voted}."
                        ),
                    )
            else:
                most_voted, vote_count = None, 0

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
                    if most_voted
                    else f"No valid votes were cast, so the Insider ({insider_name}) "
                    f"was not identified. Insider wins!"
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
    pin_secret_word(config)
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
