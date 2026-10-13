"""Launcher and engine for Bohnanza (Uwe Rosenberg, AMIGO 1997): four-player benchmark adaptation.

Original mechanics kept: the 154-card deck with the printed beanometers, fixed hand order
(plant from the front, draw to the back), two fields plus a purchasable third, the
plant / trade / plant-traded / draw turn, trades only between the active player and the
others, gifts needing consent, received cards that must be planted, bean protection, and
harvest payouts. Benchmark adaptations are listed in description.md.
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    graded_rank_rewards,
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

STATES = ("Plant", "Trade", "Resolve", "Plant_traded")


def _shared(name: str) -> Any:
    """A helper module shared by the games, loaded by path like the game modules themselves."""
    key = f"games_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, BASE_DIR.parent / f"{name}.py")
        assert spec and spec.loader
        sys.modules[key] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sys.modules[key])
    return sys.modules[key]


# Only an accept the reply commits to counts: "I accept none of these offers", "I'd accept if you add a Soy" and
# "Maybe I'll accept #1 later" accept nothing.
committed_move = _shared("committed_move")
GAME_OVER = "Game_over"

# ============================================================================
# Parsing helpers
# ============================================================================

_BEAN_WORDS = "coffee|wax|blue|chili|stink|green|soy|black-eyed|red|garden|cocoa"
_BEAN_RE = re.compile(
    rf"(?:(\d+)\s*x?\s*)?\b({_BEAN_WORDS})\b(?:\s*beans?)?(?:\s*x\s*(\d+))?"
)
_CANON = {
    "coffee": "Coffee",
    "wax": "Wax",
    "blue": "Blue",
    "chili": "Chili",
    "stink": "Stink",
    "green": "Green",
    "soy": "Soy",
    "black-eyed": "Black-eyed",
    "red": "Red",
    "garden": "Garden",
    "cocoa": "Cocoa",
}
BEAN_NAMES = set(_CANON.values())

# Command vocabulary. Small models paraphrase, so several verbs map to one command.
_VERB_MAP = {
    "plant": "plant", "sow": "plant", "put": "plant", "place": "plant", "plnt": "plant",
    "harvest": "harvest", "sell": "harvest",
    "buy": "buy",
    "offer": "offer", "trade": "offer", "swap": "offer",
    "accept": "accept", "take": "accept",
    "decline": "decline", "reject": "decline",
    "pass": "pass", "done": "pass",
}
_VERB_RE = re.compile(r"\b(" + "|".join(_VERB_MAP) + r")\b", re.I)
_FILLER_RE = re.compile(
    r"^(?:[\W_]+|\b(?:sure|okay|ok|alright|well|so|now|first|next|also|and|then|please|"
    r"i|i'?ll|i will|i'?d like to|i would like to|i want to|i'?m going to|i am going to|"
    r"let me|let'?s|my move(?: is)?|here is my move|my action(?: is)?)\b)+",
    re.I,
)
_NEG_RE = re.compile(
    r"\b(not|no|never|don'?t|won'?t|can'?t|cannot|couldn'?t|shouldn'?t|wouldn'?t|"
    r"refus\w*|declin\w*|reject\w*|without|rather than|instead of|unless|avoid\w*)\b",
    re.I,
)
_HEDGE_RE = re.compile(
    r"\b(later|next (?:turn|round|time)|when|if|maybe|perhaps|should|would|could|might|"
    r"plan(?:s|ned|ning)?|going to|thinking|consider\w*|probably|hope|hoping|wish|"
    r"eventually|someday|once|until)\b",
    re.I,
)
_PROSE_WORDS = {"my", "the", "a", "an", "card", "cards", "into", "it", "this", "that", "there", "here", "of", "some"}
_NOTHING = re.compile(r"\b(nothing|free|gift|no\s*thing|none|zero|nil)\b", re.I)
_SIDE_STRIP = re.compile(
    r"\b(my|your|his|her|their|our|the|a|an|one of|some of|some|any|these|those|this|that|"
    r"beans?|cards?|from my hand|of)\b",
    re.I,
)
_CLAUSE_CUT = re.compile(
    r"\b(because|but|if|since|so|then|also|unless|although|though|while|when|after|before|"
    r"please|thanks|which|that|want|need|would|could|later|now|only|i|you|we|they|he|she)\b",
    re.I,
)


def _norm(text: str) -> str:
    t = text.lower()
    t = re.sub(r"black[\s_]*eyed", "black-eyed", t)
    t = t.replace("chilli", "chili").replace("cacao", "cocoa")
    return t


def clean_text(text: str) -> str:
    """Strip reasoning blocks, markdown and step labels; write field ordinals as digits."""
    t = re.sub(r"<think>.*?(?:</think>|$)", " ", text, flags=re.S | re.I)
    t = t.replace("```", " ")
    t = re.sub(r"(?m)^\s*>+", " ", t)
    t = re.sub(r"[*_`|]+", " ", t)
    t = re.sub(r"(?m)^\s*(?:[-•]|\d+[.)])\s+", "", t)
    t = re.sub(r"(?i)\b(?:step|action|move|turn|response)\s*\d*\s*[:\-]\s*", " ", t)
    words = {"one": "1", "first": "1", "1st": "1", "two": "2", "second": "2", "2nd": "2", "three": "3", "third": "3", "3rd": "3"}
    t = re.sub(r"(?i)\bfield\s+(one|two|three|first|second|third|1st|2nd|3rd)\b", lambda m: "field " + words[m.group(1).lower()], t)
    t = re.sub(r"(?i)\b(?:the\s+)?(first|second|third|1st|2nd|3rd)\s+field\b", lambda m: "field " + words[m.group(1).lower()], t)
    t = re.sub(r"(?i)\b(in|into|to|on)\s+(?:the\s+)?(first|second|third)\b(?!\s+(?:card|turn|round|player|offer|one))", lambda m: f"{m.group(1)} field {words[m.group(2).lower()]}", t)
    return re.sub(r"[ \t]+", " ", t)


def parse_beans(text: str) -> List[str]:
    """'2 soy, chili x2, Black eyed bean' -> ['Soy', 'Soy', 'Chili', 'Chili', 'Black-eyed']."""
    out: List[str] = []
    for m in _BEAN_RE.finditer(_norm(text)):
        n = int(m.group(1) or m.group(3) or 1)
        out.extend([_CANON[m.group(2)]] * max(1, min(n, 20)))
    return out


class Fragment:
    """One command-sized piece of a message: a verb plus the words up to the next verb."""

    def __init__(self, prefix: str, body: str):
        self.raw = (prefix + " " + body).strip()
        body = _FILLER_RE.sub("", body).strip()
        self.text = body
        m = _VERB_RE.match(body)
        self.verb = _VERB_MAP[m.group(1).lower()] if m else None
        lead = _FILLER_RE.sub("", prefix).strip()
        # words between the previous command (or sentence start) and this verb decide negation
        self.verb_first = bool(m) and not lead
        self.negated = bool(_NEG_RE.search(lead if m else body))
        self.hedged = bool(_HEDGE_RE.search(lead + " " + body))
        toks = re.findall(r"[a-z'#-]+", _norm(body))
        self.pure = self.verb_first and len(toks) <= 6 and not (set(toks) & _PROSE_WORDS)

    @property
    def usable(self) -> bool:
        return not (self.negated or self.hedged)


_SENTENCE_SPLIT = re.compile(
    r"[;\n]|\.(?=\s|$)|,|\bthen\b|\band\b(?=\s*(?:i\b|also\b|then\b|(?:the\s+|my\s+)?(?:field\s*)?\d\b|(?:the\s+|my\s+)?(?:" + _BEAN_WORDS + r")\b|"
    r"plant|sow|put|place|harvest|sell|buy|offer|trade|accept|take|decline|reject)\b)",
    re.I,
)
_VERB_START = re.compile(r"\b(?:plant|sow|put|place|plnt|harvest|sell|buy|offer|trade|swap|accept|take|decline|reject|pass|done)\b", re.I)


def fragments(text: str) -> List[Fragment]:
    """Split a cleaned message into command fragments, in order.

    Each sentence is cut before every verb; the words in front of a verb travel with it,
    so "I don't want to harvest field 1. plant 2" keeps its negation. When a message
    states an intention in prose and then gives the bare command ("I will plant my first
    card into field 1. plant 1"), the bare command wins: for each verb, prose fragments
    are dropped whenever a pure command fragment exists.
    """
    frags: List[Fragment] = []
    for sentence in _SENTENCE_SPLIT.split(text):
        if not sentence or not sentence.strip():
            continue
        starts = [m.start() for m in _VERB_START.finditer(sentence)]
        if not starts:
            frags.append(Fragment("", sentence))
            continue
        prev_end = 0
        for i, st_ in enumerate(starts):
            end = starts[i + 1] if i + 1 < len(starts) else len(sentence)
            prefix = sentence[prev_end:st_] if i == 0 else ""
            frags.append(Fragment(prefix, sentence[st_:end]))
            prev_end = end
    frags = [f for f in frags if f.text]
    pure_verbs = {f.verb for f in frags if f.pure and f.usable}
    return [f for f in frags if not (f.verb in pure_verbs and not f.pure)]


def _continues_plant(text: str) -> bool:
    """'2', 'field 2', 'the Soy in field 2', 'Chili 2': a plant continuation without its verb."""
    low = _norm(text).strip()
    toks = low.split()
    if len(toks) > 7:
        return False
    opens_right = re.match(rf"^(?:the\s+|my\s+|in\s+|into\s+)?(?:field\s*)?(?:\d\b|(?:{_BEAN_WORDS})\b)", low)
    # "Chili is valuable" opens with a bean but names no field and is not a bare "Chili" / "Chili 2"
    return bool(opens_right) and (re.search(r"\b[1-3]\b", low) is not None or len(toks) <= 2)


def field_index(cmd: str) -> Optional[int]:
    """Field number (0-based) named in one command; explicit 'field N' / '#N' wins over a bare digit.

    Bean names and quantities ('2 chili', 'chili x2') are removed before a bare digit is read,
    so 'plant 2 chili in field 1' is field 1 and 'plant 2 chili' names no field.
    """
    low = _norm(cmd)
    m = re.search(r"\bfield\s*#?\s*([1-3])\b|#\s*([1-3])\b", low)
    if m:
        return int(m.group(1) or m.group(2)) - 1
    v = _VERB_RE.search(low)
    if v:
        low = low[v.end():]
    stripped = re.sub(rf"(\d+\s*x?\s*)?\b(?:{_BEAN_WORDS})\b(?:\s*beans?)?(\s*x\s*\d+)?", " ", low)
    stripped = re.sub(r"\b(?:into|in|to|on|field|the|my|a|it|card|cards|there|here)\b", " ", stripped)
    m = re.search(r"(?<![\d.\-])\b([1-3])\b(?![\d.])", stripped)
    return int(m.group(1)) - 1 if m else None


def _negated_at(text: str, start: int) -> bool:
    """True when the verb at `start` sits in a clause that negates it ('I do not accept #1')."""
    before = text[:start]
    clause = re.split(r"[.;!?\n]", before)[-1]
    piece = re.split(r"\b(?:but|and|so|yet|however)\b|,", clause, flags=re.I)[-1]
    tail = " ".join(piece.split()[-3:])
    conditional = re.search(r"\b(would|could|might|may)\b(?!\s+(?:like|love|be happy|be glad|want|prefer))", tail, re.I)
    return bool(_NEG_RE.search(tail) or conditional)


def _side(text: str) -> Tuple[List[str], str]:
    """Bean list for one side of an offer plus a problem code ('' / 'or' / 'empty')."""
    t = re.split(r"[.;!?\n:()]", text)[0]
    t = _SIDE_STRIP.sub(" ", t)
    m = _CLAUSE_CUT.search(t)
    if m and not _BEAN_RE.search(_norm(t[: m.start()])) and not _NOTHING.search(t[: m.start()]):
        pass  # the cut word came before any bean: keep looking past it
    else:
        t = t[: m.start()] if m else t
    if re.search(r"\bor\b", t, re.I):
        return [], "or"
    beans = parse_beans(t)
    if beans:
        return beans, ""
    if _NOTHING.search(t):
        return [], ""
    return [], "empty"


# A player name: capitalised word that is not a bean variety.
_NAME = r"((?!(?:Coffee|Wax|Blue|Chili|Stink|Green|Soy|Black|Red|Garden|Cocoa)\b)[A-Z][a-z]+)"
# One side of an offer never crosses a sentence boundary, and never holds another "offer": otherwise a lead-in
# such as "so I'm offering:" followed by "offer Karen: give Cocoa for Soy" on the next line swallowed that offer.
_BEANS_TXT = r"((?:(?!\boffer\b)[^.;!?\n])+?)"
_OFFER_PATTERNS = [
    # offer Bob: give X for Y | offer Bob X for Y | offer: give X for Y | I offer my X for your Y | offering: ...
    # ("offering" is read as one word: the name group is case-insensitive, so "offering" once gave the player "ing")
    re.compile(
        rf"\boffer(?:s|ing)?\b\s*(?:to\s+)?(?:{_NAME}\s*)?[:\-]?\s*(?:i\s+)?(?:will\s+|can\s+|could\s+)?(?:give\s+(?:you\s+|{_NAME}\s+)?)?{_BEANS_TXT}\s+(?:for|in exchange for|in return for|to get|->|→)\s+{_BEANS_TXT}(?=(?:\boffer\b|[.;!?\n]|$))",
        re.I,
    ),
    # trade/swap X for Y (with Bob)
    re.compile(
        rf"\b(?:trade|swap|exchange)\s+{_BEANS_TXT}\s+(?:for|against)\s+{_BEANS_TXT}(?:\s+with\s+{_NAME})?(?=(?:\boffer\b|[.;!?\n]|$))",
        re.I,
    ),
    # give Bob my X (for his Y)
    re.compile(
        rf"\bgive\s+{_NAME}\s+{_BEANS_TXT}(?:\s+(?:for|in exchange for|in return for)\s+{_BEANS_TXT})?(?=(?:\boffer\b|[.;!?\n]|$))",
        re.I,
    ),
]


_NOT_A_TARGET = {
    "to", "the", "a", "an", "you", "me", "everyone", "anyone", "all", "someone", "somebody", "i", "my", "your",
    "give", "for", "nothing", "offer", "trade", "swap", "some", "any", "this", "that", "these", "those", "his",
    "her", "their", "our",
}


def parse_offers(text: str) -> Tuple[List[Tuple[Optional[str], List[str], List[str]]], List[str]]:
    """Return ([(target name or None, give, get)], problems) from a free-text message."""
    offers = []
    problems: List[str] = []
    seen = set()
    for pat in _OFFER_PATTERNS:
        for m in pat.finditer(text):
            if any(a <= m.start() < b for a, b in seen):
                continue
            if _negated_at(text, m.start()):
                continue
            groups = list(m.groups())
            if pat is _OFFER_PATTERNS[0]:
                target = groups[0] or groups[1]
                give_txt, get_txt = groups[2], groups[3]
            elif pat is _OFFER_PATTERNS[1]:
                give_txt, get_txt, target = groups[0], groups[1], groups[2]
            else:
                target, give_txt, get_txt = groups[0], groups[1], groups[2] or "nothing"
            if target and target.lower() in _NOT_A_TARGET:
                target = None
            give, p1 = _side(give_txt)
            get, p2 = _side(get_txt)
            if p1 == "or" or p2 == "or":
                problems.append("an offer must name exactly what you give and what you get (no 'or')")
                continue
            if p1 == "empty" and p2 == "empty":
                continue
            if p1 == "empty" or p2 == "empty":
                problems.append("could not read one side of your offer; write 'offer <Name>: give <cards> for <cards>' (use 'nothing' for a gift)")
                continue
            if not give and not get:
                continue
            seen.add((m.start(), m.end()))
            offers.append((target, give, get))
    return offers, list(dict.fromkeys(problems))


_ACCEPT_STOP = re.compile(r"\b(but|reject|decline|not|except|other than|besides|apart from|excluding)\b|[.;!?\n]", re.I)


def parse_accepts(text: str) -> Dict[str, Any]:
    """Which offers a message accepts: ids, 'all' (with exceptions), offers from named players, or a bare accept.
    Only accepts the message commits to count (committed_move)."""
    out: Dict[str, Any] = {"all": False, "ids": [], "except": [], "names": [], "bare": False}
    if re.fullmatch(r"\W*(?:yes|ok|okay|sure|deal|agreed|fine|yep|yeah)\W*", text, re.I):
        out["bare"] = True
        return out
    for mention in committed_move.mentions(text, {"accept": r"\b(accept(?:s|ing)?|take|taking)\b"}):
        if not mention.committed:
            continue  # negated, hypothetical, conditional, for later, or "accept none / nothing / neither"
        m = re.compile(r"\b(accept(?:s|ing)?|take|taking)\b", re.I).match(text, mention.start)
        if m is None:
            continue
        if m.group(1).lower() in ("take", "taking") and not re.search(r"\b(offer|deal|it|that)\b|'s\b|#\s*\d", text[m.end(): m.end() + 40], re.I):
            continue  # 'take' only when it refers to an offer
        head = text[m.end():]
        stop = _ACCEPT_STOP.search(head)
        clause = head[: stop.start()] if stop else head
        rest = head[stop.end():] if stop else ""
        if re.search(r"\b(all|everything|every offer|both|any)\b", clause, re.I):
            out["all"] = True
            exc = re.search(r"\b(?:except|but not|other than|besides|apart from|excluding)\b(.*)", head, re.I)
            if exc:
                out["except"] += [int(x) for x in re.findall(r"#?\s*(\d+)", exc.group(1)[:60])]
            continue
        for lo, hi in re.findall(r"#?\s*(\d+)\s*(?:-|–|to|through)\s*#?\s*(\d+)", clause):
            out["ids"] += list(range(int(lo), int(hi) + 1))
        clause_wo_ranges = re.sub(r"#?\s*\d+\s*(?:-|–|to|through)\s*#?\s*\d+", " ", clause)
        for x in re.finditer(r"(?:#\s*|offers?\s*#?\s*)(\d+)|\b(\d+)\b(?!\s*(?:" + _BEAN_WORDS + r"|coins?|cards?|beans?))", clause_wo_ranges, re.I):
            out["ids"].append(int(x.group(1) or x.group(2)))
        for n in re.findall(rf"{_NAME}'s\b|\bfrom\s+{_NAME}", clause):
            name = n[0] or n[1]
            if name:
                out["names"].append(name)
        for o in re.findall(r"\b(first|second|third|fourth|last)\s+(?:offer|one)\b", clause, re.I):
            out.setdefault("ordinals", []).append(o.lower())
        if not out["ids"] and not out["names"] and not out["all"]:
            out["bare"] = True
    out["ids"] = list(dict.fromkeys(out["ids"]))
    return out



# ============================================================================
# Evaluator
# ============================================================================


class BohnanzaEvaluator(SocialGameEndEvaluator):
    def __call__(
        self, turn_number: int, messages: List[Tuple[str, Message]], **kwargs: Any
    ) -> List[Tuple[str, Tuple[Tuple[str, int | float | bool], str]]]:
        env = kwargs.get("env")
        if env and not env.internal_state.get("game_over") and turn_number >= self.max_turn_number:
            env._end_game(f"turn cap of {self.max_turn_number} reached")  # score what was played
        if env and env.internal_state.get("game_over", False):
            scores = env.internal_state.get("final_scores", {})
            reason = env.internal_state.get("end_reason", "Game over")
            response: List[Tuple[str, Tuple[Tuple[str, int | float | bool], str]]] = [
                ("environment", (("terminated", True), reason))
            ]
            for idx, name in enumerate(env.agents):
                response.append(
                    (f"agent_{idx + 1}", (("complete_rating", scores.get(name, 0.0)), ""))
                )
            return response
        if turn_number >= self.max_turn_number:
            return [("environment", (("terminated", True), "Timeout"))]
        return [("environment", (("terminated", False), ""))]


# ============================================================================
# Action handler
# ============================================================================


class BohnanzaActionHandler(ActionHandler):
    def handle_action(
        self, env: SocialDeductionGame, agent_name: str, action: AgentAction
    ) -> None:
        if not isinstance(env, BohnanzaEnv) or env.internal_state.get("game_over"):
            return
        state = env.current_state
        if action.action_type not in ("action", "speak"):
            if state == "Resolve" and agent_name == env.active_player():
                # A failed call in Resolve declines the open offers, as any other non-accept does, said publicly
                env._announce(f"{agent_name} gave no answer, so the open offers are declined.")
            return
        text = clean_text(action.argument or "")
        if state == "Plant" and agent_name == env.active_player():
            env.handle_plant_phase(agent_name, text)
        elif state == "Trade":
            env.handle_trade_message(agent_name, text)
        elif state == "Resolve" and agent_name == env.active_player():
            env.handle_resolve(agent_name, text)
        elif state == "Plant_traded" and env.internal_state["pending"].get(agent_name):
            env.handle_plant_pending(agent_name, text)

    def get_action_instruction(self, env: SocialDeductionGame, agent_name: str) -> str:
        if not isinstance(env, BohnanzaEnv) or not env.internal_state.get("hands"):
            return ""
        return env.instruction_for(agent_name)


# ============================================================================
# Environment
# ============================================================================


class BohnanzaEnv(SocialDeductionGame):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(action_handler=BohnanzaActionHandler(), **kwargs)

    # ---- setup -------------------------------------------------------------

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
        cfg = self._config
        self._rng = random.Random(f"{cfg.get('seed')}-bohnanza")
        self.beans: Dict[str, Dict[str, Any]] = {b["name"]: b for b in cfg["beans"]}
        removed = set(cfg.get("removed_varieties", []))
        deck: List[str] = []
        for b in cfg["beans"]:
            if b["name"] not in removed:
                deck.extend([b["name"]] * int(b["count"]))
        self._rng.shuffle(deck)
        names = list(self.agents)
        self.internal_state = {
            "deck": deck,
            "total_cards": len(deck),
            "discard": [],
            "coin_cards": 0,
            "hands": {n: [] for n in names},
            "fields": {n: [[] for _ in range(int(cfg.get("starting_fields", 2)))] for n in names},
            "coins": {n: 0 for n in names},
            "exhaustions": 0,
            "deck_out": False,
            "ending_after_turn": False,
            "active_idx": 0,
            "round": 1,
            "market": [],
            "offers": {},
            "next_offer_id": 1,
            "pending": {n: [] for n in names},
            "planted_first": False,
            "plants_this_phase": 0,
            "fallbacks": 0,
            "trades": 0,
            "game_over": False,
            "final_scores": {},
            "end_reason": "",
        }
        hand_size = int(cfg.get("hand_size", 5))
        for _ in range(hand_size):
            for n in names:
                card = self._draw()
                if card:
                    self.internal_state["hands"][n].append(card)
        self._announce(self._table_summary(prefix=f"Round 1, {names[0]}'s turn."))
        self._update_action_mask()
        for n in names:
            obs[n].action_instruction = self.instruction_for(n)
            obs[n].available_actions = self._get_available_actions(n)
        return obs

    # ---- small accessors ---------------------------------------------------

    def active_player(self) -> str:
        agents = list(self.agents)
        return agents[self.internal_state.get("active_idx", 0) % len(agents)]

    def _announce(self, msg: str, receivers: List[str] | None = None) -> None:
        self.recv_message("Environment", SimpleMessage(message=f"[Game] {msg}"), receivers=receivers)

    def coins_for(self, bean: str, n: int) -> int:
        meter = self.beans[bean]["beanometer"]
        best = 0
        for level, need in enumerate(meter, start=1):
            if need is not None and n >= need:
                best = level
        return best

    def _field_str(self, f: List[str]) -> str:
        return f"{f[0]} x{len(f)}" if f else "empty"

    def _table_summary(self, prefix: str = "") -> str:
        st = self.internal_state
        parts = []
        for n in self.agents:
            fields = " | ".join(
                f"{i + 1}: {self._field_str(f)}" for i, f in enumerate(st["fields"][n])
            )
            parts.append(f"{n}: {st['coins'][n]} coins, hand {len(st['hands'][n])}, fields [{fields}]")
        return (prefix + " " if prefix else "") + "Table: " + "; ".join(parts) + "."

    # ---- deck --------------------------------------------------------------

    def _draw(self) -> Optional[str]:
        st = self.internal_state
        if st["deck_out"]:
            return None
        if not st["deck"]:
            st["exhaustions"] += 1
            limit = int(self._config.get("deck_exhaustions_to_end", 3))
            if st["exhaustions"] >= limit or not st["discard"]:
                st["deck_out"] = True
                return None
            st["deck"] = st["discard"]
            st["discard"] = []
            self._rng.shuffle(st["deck"])
            self._announce(
                f"The draw pile ran out ({st['exhaustions']}/{limit}); the discard pile was shuffled into a new draw pile."
            )
        return st["deck"].pop()

    # ---- fields ------------------------------------------------------------

    def _auto_field_of(self, name: str, hand_card: Optional[str]) -> Optional[int]:
        """Index of the field that received the last planted bean (used for 'plant ... there')."""
        fields = self.internal_state["fields"][name]
        return max(range(len(fields)), key=lambda i: len(fields[i])) if any(fields) else None

    def _auto_field(self, name: str, bean: str) -> Optional[int]:
        fields = self.internal_state["fields"][name]
        for i, f in enumerate(fields):
            if f and f[0] == bean:
                return i
        for i, f in enumerate(fields):
            if not f:
                return i
        return None

    def _harvestable(self, name: str, idx: int) -> Tuple[bool, str]:
        fields = self.internal_state["fields"][name]
        if not (0 <= idx < len(fields)):
            return False, f"you have no field {idx + 1}"
        f = fields[idx]
        if not f:
            return False, f"field {idx + 1} is empty"
        if len(f) == 1 and any(len(g) >= 2 for j, g in enumerate(fields) if j != idx):
            return False, f"bean protection: field {idx + 1} holds a single bean while another field holds 2 or more"
        return True, ""

    def _harvest(self, name: str, idx: int, force: bool = False) -> Tuple[bool, str]:
        st = self.internal_state
        if not force:
            ok, why = self._harvestable(name, idx)
            if not ok:
                return False, why
        f = st["fields"][name][idx]
        if not f:
            return False, f"field {idx + 1} is empty"
        bean, n = f[0], len(f)
        coins = self.coins_for(bean, n)
        st["coins"][name] += coins
        st["coin_cards"] += coins
        st["discard"].extend([bean] * (n - coins))
        st["fields"][name][idx] = []
        return True, f"{name} harvests {n} {bean} from field {idx + 1} for {coins} coin(s) (total {st['coins'][name]})"

    def _plant(self, name: str, bean: str, idx: Optional[int]) -> Tuple[bool, str]:
        fields = self.internal_state["fields"][name]
        if idx is None:
            idx = self._auto_field(name, bean)
            if idx is None:
                return False, f"no field can take {bean}: name a field to harvest and replant"
        if not (0 <= idx < len(fields)):
            return False, f"you have no field {idx + 1}"
        f = fields[idx]
        notes = []
        if f and f[0] != bean:
            ok, msg = self._harvest(name, idx)
            if not ok:
                return False, f"cannot plant {bean} in field {idx + 1}: {msg}"
            notes.append(msg)
            f = fields[idx]
        f.append(bean)
        notes.append(f"{name} plants {bean} in field {idx + 1} (now {bean} x{len(f)})")
        return True, "; ".join(notes)

    def _forced_plant(self, name: str, bean: str) -> str:
        """Engine fallback when a card must be planted and the player gave no legal instruction."""
        st = self.internal_state
        idx = self._auto_field(name, bean)
        if idx is None:
            fields = st["fields"][name]
            candidates = [i for i in range(len(fields)) if self._harvestable(name, i)[0]]
            if not candidates:
                candidates = [i for i, f in enumerate(fields) if f]
            idx = max(
                candidates,
                key=lambda i: (self.coins_for(fields[i][0], len(fields[i])), len(fields[i]), -i),
            )
            ok, msg = self._harvest(name, idx, force=True)
            self._announce(msg)
        ok, msg = self._plant(name, bean, idx)
        st["fallbacks"] += 1
        return msg + " [automatic placement]"

    def _buy_field(self, name: str) -> Tuple[bool, str]:
        st = self.internal_state
        cost = int(self._config.get("third_field_cost", 3))
        if len(st["fields"][name]) >= int(self._config.get("max_fields", 3)):
            return False, "you already own the maximum number of fields"
        if st["coins"][name] < cost:
            return False, f"a third field costs {cost} coins; you have {st['coins'][name]}"
        st["coins"][name] -= cost
        st["coin_cards"] -= cost
        st["discard"].extend(self._coin_cards_back(cost))
        st["fields"][name].append([])
        return True, f"{name} buys a third field for {cost} coins (now {st['coins'][name]} coins)"

    def _coin_cards_back(self, k: int) -> List[str]:
        # Coins are bean cards turned face down; spending them returns cards to the discard pile.
        # Which variety they were is irrelevant to play, so use the most common variety in play.
        removed = set(self._config.get("removed_varieties", []))
        variety = max((b for b in self._config["beans"] if b["name"] not in removed), key=lambda b: b["count"])["name"]
        return [variety] * k

    # ---- shared command helpers ----------------------------------------------

    def _note(self, name: str, msg: str) -> None:
        """Private notice to one player (a rejected command; never broadcast)."""
        self.internal_state["rejected"] = self.internal_state.get("rejected", 0) + 1
        self._announce(msg, receivers=[name])

    def _run_harvest(self, name: str, frag: "Fragment") -> None:
        idx = field_index(frag.text)
        if idx is None:
            self._note(name, "'harvest' needs a field number, e.g. 'harvest 2'.")
            return
        ok, msg = self._harvest(name, idx)
        if ok:
            self._announce(msg)
        else:
            self._note(name, f"Cannot harvest: {msg}.")

    def _run_buy(self, name: str, frag: "Fragment") -> None:
        if not re.search(r"\bfield\b", frag.text, re.I) and not frag.pure:
            return
        ok, msg = self._buy_field(name)
        if ok:
            self._announce(msg)
        else:
            self._note(name, f"Cannot buy a field: {msg}.")

    def _side_commands(self, name: str, frags: List["Fragment"]) -> None:
        """'harvest <field>' and 'buy field' from any message a player sends (rulebook: any time).

        Only verb-first, non-negated, non-hedged fragments count, so prose such as
        "I plan to harvest field 1 later" or "I won't buy a field" does nothing.
        """
        for f in frags:
            if not (f.verb_first and f.usable):
                continue
            if f.verb == "harvest":
                self._run_harvest(name, f)
            elif f.verb == "buy":
                self._run_buy(name, f)

    # ---- Plant phase (active player, one message) --------------------------

    def handle_plant_phase(self, name: str, text: str) -> None:
        st = self.internal_state
        hand = st["hands"][name]
        plants = 0
        last_verb = None
        last_field: Optional[int] = None
        for f in fragments(text):
            if f.verb == "pass" and f.pure:
                break
            if not f.usable:
                continue
            verb = f.verb
            if verb is None:
                # "plant 1 and 2" / "... and the Soy in field 2": a short verbless piece that opens
                # with a field, a number or a bean continues the previous plant command
                if last_verb == "plant" and _continues_plant(f.text):
                    verb = "plant"
                else:
                    continue
            if verb == "buy":
                self._run_buy(name, f)
                last_field = field_index(f.text)
            elif verb == "harvest":
                self._run_harvest(name, f)
                last_field = field_index(f.text)
            elif verb == "plant":
                repeats = 2 if re.search(r"\b(both|two cards|2 cards|both cards)\b", f.text, re.I) else 1
                for _ in range(repeats):
                    if plants >= 2:
                        self._note(name, "At most two cards may be planted in the Plant phase.")
                        break
                    if not hand:
                        self._note(name, "Your hand is empty.")
                        break
                    beans = parse_beans(f.text)
                    named = [b for b in beans if b != hand[0]]
                    idx = field_index(f.text)
                    if idx is None and last_field is not None and re.search(r"\b(there|that field|it)\b", f.text, re.I):
                        idx = last_field
                    if named and hand[0] not in beans:
                        # "plant Soy in field 2" with Chili in front: only the front card may be planted. It goes
                        # into the field the reply names when that is legal (the engine does not move it to a
                        # field it thinks was meant), else the usual placement applies.
                        self._note(name, f"Only the front card ({hand[0]}) can be planted in this phase; {named[0]} was ignored.")
                    ok, msg = self._plant(name, hand[0], idx)
                    if ok:
                        hand.pop(0)
                        plants += 1
                        self._announce(msg)
                        last_field = idx if idx is not None else self._auto_field_of(name, hand_card=None)
                    else:
                        self._note(name, f"Cannot plant {hand[0]}: {msg}.")
            last_verb = verb
        if plants == 0 and hand:
            msg = self._forced_plant(name, hand.pop(0))
            self._announce(f"{name} did not plant a legal card; {msg}")
        st["planted_first"] = True
        st["plants_this_phase"] = plants

    # ---- Trade phase ---------------------------------------------------------

    def _sources(self, name: str) -> List[str]:
        st = self.internal_state
        cards = list(st["hands"][name])
        if name == self.active_player():
            cards += st["market"]
        return cards

    def _has_cards(self, name: str, beans: List[str]) -> bool:
        pool = self._sources(name)
        for b in beans:
            if b in pool:
                pool.remove(b)
            else:
                return False
        return True

    def _take_cards(self, name: str, beans: List[str]) -> None:
        """Remove beans from a player's tradable cards (face-up cards first for the active player)."""
        st = self.internal_state
        for b in beans:
            if name == self.active_player() and b in st["market"]:
                st["market"].remove(b)
            else:
                st["hands"][name].remove(b)

    def _in_play(self) -> set:
        removed = set(self._config.get("removed_varieties", []))
        return {b["name"] for b in self._config["beans"] if b["name"] not in removed}

    def _open_offers_to(self, name: str) -> Dict[int, Dict[str, Any]]:
        return {oid: o for oid, o in self.internal_state["offers"].items() if o["to"] == name and o["open"]}

    def _apply_accepts(self, name: str, text: str) -> None:
        acc = parse_accepts(text)
        mine = self._open_offers_to(name)
        ids: List[int] = []
        if acc["all"]:
            ids += [oid for oid in mine if oid not in acc["except"]]
        ids += acc["ids"]
        for who in acc["names"]:
            target = self._match_name(who)
            ids += [oid for oid, o in mine.items() if o["from"] == target]
        if acc.get("ordinals"):
            order = list(mine)
            for o in acc["ordinals"]:
                k = {"first": 0, "second": 1, "third": 2, "fourth": 3, "last": len(order) - 1}[o]
                if 0 <= k < len(order):
                    ids.append(order[k])
        if acc["bare"] and not ids:
            if len(mine) == 1:
                ids = list(mine)
            elif mine:
                self._note(name, "Say which offer you accept, e.g. 'accept #" + " #".join(str(i) for i in mine) + "' or 'accept all'.")
            else:
                self._note(name, "There is no open offer addressed to you.")
        for oid in dict.fromkeys(ids):
            self._accept_offer(name, oid)

    def handle_trade_message(self, name: str, text: str) -> None:
        st = self.internal_state
        active = self.active_player()
        self._side_commands(name, fragments(text))
        self._apply_accepts(name, text)
        offers, problems = parse_offers(text)
        for prob in problems:
            self._note(name, prob.capitalize() + ".")
        # "<cards>" left in: the instruction's format copied without filling it in, which posts nothing.
        if not offers and not problems and re.search(r"\b(offer|trade|swap|give)\b", text, re.I) and (_BEAN_RE.search(_norm(text)) or "<" in text):
            self._note(name, "No offer was posted. Format: 'offer <Name>: give <cards> for <cards>' (write 'nothing' for a gift).")
        for target, give, get in offers:
            if name == active:
                if not target:
                    self._note(name, "An offer must name the player it is addressed to, e.g. 'offer Bob: give Chili for Soy'.")
                    continue
                to = self._match_name(target)
                if to == name:
                    self._note(name, "You cannot trade with yourself.")
                    continue
                if to is None:
                    self._note(name, f"Unknown player '{target}'. Players: {', '.join(n for n in self.agents if n != name)}.")
                    continue
            else:
                to = active
                if target and self._match_name(target) not in (None, active, name):
                    self._note(name, f"Only trades with {active} are allowed this turn.")
                    continue
            self._record_offer(name, to, give, get)

    def _match_name(self, token: str) -> Optional[str]:
        for n in self.agents:
            if n.lower() == token.lower():
                return n
        for n in self.agents:
            if n.lower().startswith(token.lower()) and len(token) >= 3:
                return n
        return None

    def _record_offer(self, frm: str, to: str, give: List[str], get: List[str]) -> None:
        st = self.internal_state
        if not give and not get:
            return
        unknown = [b for b in give + get if b not in self._in_play()]
        if unknown:
            self._note(frm, f"{', '.join(sorted(set(unknown)))} is not in this game's deck; your offer was not posted.")
            return
        if not self._has_cards(frm, give):
            # Private: the table must not learn what a player holds from a rejected offer.
            self._note(frm, f"Your offer was not posted: you do not hold {', '.join(give)}.")
            return
        oid = st["next_offer_id"]
        st["next_offer_id"] += 1
        st["offers"][oid] = {"from": frm, "to": to, "give": give, "get": get, "open": True}
        self._announce(f"Offer #{oid}: {frm} gives {self._fmt(give)} to {to} for {self._fmt(get)}. {to} may 'accept #{oid}'.")

    @staticmethod
    def _fmt(beans: List[str]) -> str:
        return ", ".join(beans) if beans else "nothing"

    def _accept_offer(self, acceptor: str, oid: int) -> None:
        st = self.internal_state
        o = st["offers"].get(oid)
        if not o or not o["open"]:
            self._note(acceptor, f"Offer #{oid} is not open.")
            return
        if o["to"] != acceptor:
            self._note(acceptor, f"Offer #{oid} is not addressed to you.")
            return
        if not self._has_cards(o["from"], o["give"]):
            o["open"] = False
            self._announce(f"Offer #{oid} can no longer be completed (cards already traded away).")
            return
        if not self._has_cards(o["to"], o["get"]):
            self._note(acceptor, f"You cannot take offer #{oid}: you do not hold {', '.join(o['get'])}.")
            return
        self._take_cards(o["from"], o["give"])
        self._take_cards(o["to"], o["get"])
        st["pending"][o["to"]].extend(o["give"])
        st["pending"][o["from"]].extend(o["get"])
        o["open"] = False
        st["trades"] += 1
        self._announce(
            f"Trade #{oid} done: {o['from']} gives {self._fmt(o['give'])} to {o['to']} for {self._fmt(o['get'])}. Received cards must be planted."
        )

    def handle_resolve(self, name: str, text: str) -> None:
        self._side_commands(name, fragments(text))
        before = len(self._open_offers_to(name))
        self._apply_accepts(name, text)
        offers, _ = parse_offers(text)
        if offers:
            self._note(name, "New offers cannot be made now; only 'accept #<id>' or 'pass'.")
        if before and len(self._open_offers_to(name)) == before:
            self._announce(f"{name} declines the open offers.")

    # ---- Plant_traded phase ------------------------------------------------

    def handle_plant_pending(self, name: str, text: str) -> None:
        st = self.internal_state
        pending = st["pending"][name]
        for f in fragments(text):
            if not f.usable:
                continue
            if f.verb == "harvest":
                self._run_harvest(name, f)
                continue
            if f.verb == "buy":
                self._run_buy(name, f)
                continue
            if f.verb not in (None, "plant"):
                continue
            beans = parse_beans(f.text)
            if f.verb is None and not beans:
                continue
            bean = next((b for b in beans if b in pending), None)
            if bean is None:
                if beans:
                    self._note(name, f"You have no {beans[0]} to plant; cards to plant: {self._fmt(pending)}.")
                    continue
                if not pending:
                    continue
                bean = pending[0]
            ok, msg = self._plant(name, bean, field_index(f.text))
            if ok:
                pending.remove(bean)
                self._announce(msg)
            else:
                self._note(name, f"Cannot plant {bean}: {msg}.")
        st["planted_pending"] = st.get("planted_pending", set()) | {name}

    def _flush_pending(self, name: str) -> None:
        st = self.internal_state
        while st["pending"][name]:
            bean = st["pending"][name].pop(0)
            self._announce(self._forced_plant(name, bean))

    # ---- turn flow -----------------------------------------------------------

    def _on_exit(self, state: str) -> None:
        st = self.internal_state
        if state == "Plant":
            name = self.active_player()
            if not st["planted_first"] and st["hands"][name]:
                self._announce(f"{name} did not plant; " + self._forced_plant(name, st["hands"][name].pop(0)))
        elif state == "Plant_traded":
            for n in self.agents:
                self._flush_pending(n)
            self._end_of_turn()

    def _on_enter(self, state: str) -> bool:
        """Apply entry effects; return True when someone has to act in this state."""
        st = self.internal_state
        if state == "Plant":
            st["planted_first"] = False
            st["plants_this_phase"] = 0
            self._announce(self._table_summary(prefix=f"Round {st['round']}, {self.active_player()}'s turn."))
            return True
        if state == "Trade":
            active = self.active_player()
            st["offers"] = {}
            st["next_offer_id"] = 1
            st["market"] = []
            for _ in range(2):
                card = self._draw()
                if card is None:
                    st["ending_after_turn"] = True
                    break
                st["market"].append(card)
            if st["ending_after_turn"]:
                self._announce("The draw pile is exhausted: this is the last turn; the game ends after the traded cards are planted.")
            self._announce(f"Face-up cards for {active}: {self._fmt(st['market'])}. Trading is open; only {active} may trade with the others.")
            return True
        if state == "Resolve":
            active = self.active_player()
            return any(o["open"] and o["to"] == active for o in st["offers"].values())
        if state == "Plant_traded":
            active = self.active_player()
            if st["market"]:
                st["pending"][active].extend(st["market"])
                st["market"] = []
            st["offers"] = {}
            st["planted_pending"] = set()
            todo = [n for n in self.agents if st["pending"][n]]
            if todo:
                self._announce("Cards to plant now: " + "; ".join(f"{n}: {self._fmt(st['pending'][n])}" for n in todo) + ".")
            return bool(todo)
        return False

    def _end_of_turn(self) -> None:
        st = self.internal_state
        name = self.active_player()
        if st["ending_after_turn"]:
            self._end_game("the draw pile ran out")
            return
        drawn = []
        for _ in range(3):
            card = self._draw()
            if card is None:
                break
            drawn.append(card)
            st["hands"][name].append(card)  # a drawn card is in the hand even if the game ends now
        if drawn:
            self._announce(f"{name} draws {len(drawn)} card(s) (hand {len(st['hands'][name])}).")
            self._announce(f"You drew: {self._fmt(drawn)}.", receivers=[name])
        if len(drawn) < 3:
            self._end_game("the draw pile ran out")
            return
        agents = list(self.agents)
        st["active_idx"] = (st["active_idx"] + 1) % len(agents)
        if st["active_idx"] == 0:
            st["round"] += 1
            if st["round"] > int(self._config.get("max_rounds", 6)):
                self._end_game(f"{self._config.get('max_rounds', 6)} rounds are complete")

    def _end_game(self, why: str) -> None:
        st = self.internal_state
        if st["game_over"]:
            return
        notes = []
        for n in self.agents:
            for i, f in enumerate(st["fields"][n]):
                if f:
                    ok, msg = self._harvest(n, i, force=True)
                    notes.append(msg)
        coins = dict(st["coins"])
        # Rio Grande tie-break: equal coins are ordered by cards in hand; players still equal share a fate.
        score = {n: coins[n] * 100 + len(st["hands"][n]) for n in self.agents}
        final = graded_rank_rewards(score)
        ranking = sorted(self.agents, key=lambda n: -score[n])
        top, second = ranking[0], ranking[1]
        reason = (
            f"Game over: {why}. Final harvest: " + ("; ".join(notes) if notes else "nothing to harvest") + ". "
            f"Coins (cards in hand): " + ", ".join(f"{n} {coins[n]} ({len(st['hands'][n])})" for n in ranking) + ". "
            + (
                f"{top} wins with {coins[top]} coins."
                if coins[top] > coins[second]
                else (
                    f"{top} wins the tie on coins with more cards in hand."
                    if score[top] > score[second]
                    else "The top players tie."
                )
            )
        )
        reason += f" (automatic placements: {st['fallbacks']}, rejected commands: {st.get('rejected', 0)}, trades: {st['trades']})"
        st["game_over"] = True
        st["final_scores"] = final
        st["end_reason"] = reason
        self._announce(reason)
        self.current_state = GAME_OVER

    def _perform_transition_state(self) -> None:
        st = self.internal_state
        cur = self.current_state
        if cur == GAME_OVER or st.get("game_over"):
            return
        transition = self._config.get("state_transition", {})
        while True:
            self._on_exit(cur)
            if st["game_over"]:
                return
            nxt = transition.get(cur, "Plant")
            needs_action = self._on_enter(nxt)
            if st["game_over"]:
                return
            cur = nxt
            if needs_action:
                break
        self.current_state = cur
        self._state_turn_count[cur] = 0
        if hasattr(self, "_round_robin_idx"):
            self._round_robin_idx = 0
        self._announce(f"Entering state: {cur}")

    def _update_action_mask(self) -> None:
        super()._update_action_mask()
        st = self.internal_state
        agents = list(self.agents)
        if not st or st.get("game_over") or self.current_state == GAME_OVER:
            self.action_mask = [False] * len(agents)
            return
        active = st["active_idx"] % len(agents)
        if self.current_state in ("Plant", "Resolve"):
            self.action_mask = [i == active for i in range(len(agents))]
        elif self.current_state == "Trade":
            k = getattr(self, "_state_turn_count", {}).get("Trade", 0)
            idx = (active + k) % len(agents)
            self.action_mask = [i == idx for i in range(len(agents))]
        elif self.current_state == "Plant_traded":
            self.action_mask = [bool(st["pending"].get(n)) for n in agents]

    def _should_transition_state(self) -> bool:
        if self.current_state == GAME_OVER:
            return False
        return super()._should_transition_state()

    # ---- prompts -------------------------------------------------------------

    def _status_block(self, name: str) -> str:
        st = self.internal_state
        fields = " | ".join(
            f"{i + 1}: {self._field_str(f)}" + (f" (worth {self.coins_for(f[0], len(f))} coin(s) now)" if f else "")
            for i, f in enumerate(st["fields"][name])
        )
        hand = st["hands"][name]
        return (
            f"[Your status] Coins: {st['coins'][name]}. Fields: {fields}. "
            f"Hand, front to back ({len(hand)} cards): {self._fmt(hand)}."
        )

    def instruction_for(self, name: str) -> str:
        st = self.internal_state
        if st.get("game_over"):
            return "The game is over."
        active = self.active_player()
        status = self._status_block(name)
        state = self.current_state
        hand = st["hands"][name]
        cost = int(self._config.get("third_field_cost", 3))
        can_buy = st["coins"][name] >= cost and len(st["fields"][name]) < int(self._config.get("max_fields", 3))
        buy_hint = f" 'buy field' ({cost} coins) is also allowed." if can_buy else ""
        if state == "Plant":
            if name != active:
                return f"{status} Waiting for {active} to plant."
            # Format only, no field filled in: a concrete example ('plant 1') is copied verbatim by small
            # models and uproots field 1 whenever the front card belongs elsewhere (rule audit, row 1).
            second = f" A second 'plant <field number>' also plants the next card ({hand[1]})." if len(hand) > 1 else ""
            return (
                f"{status} Your turn. Reply with commands only. 'plant <field number>' plants your front card ({hand[0] if hand else 'none'}) in the field you name.{second}"
                " The number is the FIELD. Planting into a field that holds another variety harvests that field first (bean protection applies)."
                " If your reply plants no card legally, the engine plants your front card for you (see the rules)."
                " Write 'harvest <field>' only if you want to cash that field in right now." + buy_hint
            )
        if state == "Trade":
            mine = self._open_offers_to(name)
            open_txt = ""
            if mine:
                lines = "; ".join(f"#{oid}: {o['from']} gives you {self._fmt(o['give'])} for " + (f"your {self._fmt(o['get'])}" if o["get"] else "nothing (gift)") for oid, o in mine.items())
                open_txt = f" Offers addressed to you: {lines}. Reply 'accept #<id>' to take one."
            # Formats only, no cards or players filled in: the old examples (offer the next seat a face-up card
            # for Soy; give your front card for the first face-up card; accept the first offer) were copied
            # verbatim and mostly asked for cards the copier had no field for.
            if name == active:
                return (
                    f"{status} Your face-up cards: {self._fmt(st['market'])}; whatever you do not trade away you must plant."
                    " Say briefly what you need, then list offers one per line in the form 'offer <Name>: give <cards> for <cards>'"
                    " (the cards you give, then the cards you get; write 'nothing' for a gift). You may give any card in your hand."
                    " Others answer with their own offers; you accept or decline those in the next step."
                    " Do not write the words harvest or buy unless you want to do that right now." + buy_hint
                )
            return (
                f"{status} {active} shows: {self._fmt(st['market'])}.{open_txt}"
                f" You may only trade with {active}. Say briefly what you need, then propose in the form 'offer: give <cards> for <cards>'"
                f" (the cards you give from your hand, then the cards you get from {active}'s face-up cards or hand; write 'nothing' for a gift, e.g. to take a card {active} wants gone)."
                " Any card you receive must be planted right away. Do not write the words harvest or buy unless you want to do that right now." + buy_hint
            )
        if state == "Resolve":
            if name != active:
                return f"{status} Waiting for {active} to answer the offers."
            mine = self._open_offers_to(name)
            lines = "; ".join(f"#{oid}: {o['from']} gives you {self._fmt(o['give'])} for " + (f"your {self._fmt(o['get'])}" if o["get"] else "nothing (gift)") for oid, o in mine.items())
            return (
                f"{status} Open offers to you: {lines}. Reply 'accept #<id>' naming each offer you take (several ids may go in one reply), 'accept all', or 'pass'; anything else declines all."
                " Face-up cards you keep must be planted next."
            )
        if state == "Plant_traded":
            pend = st["pending"].get(name, [])
            if not pend:
                return f"{status} Waiting for others to plant the cards they received."
            # Format only (rule audit, row 1): the old example put card i in field i whatever the fields held,
            # and "harvest first to make room" invited harvests that planting does by itself.
            return (
                f"{status} You must plant now: {self._fmt(pend)}. Reply with one 'plant <card> in field <number>' per card;"
                " planting into a field that holds another variety harvests that field first (bean protection applies)."
                " Write 'harvest <field>' only if you want to cash that field in right now. Cards you leave unplanted are placed for you, as the rules describe." + buy_hint
            )
        return status


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
) -> BohnanzaEnv:
    return BohnanzaEnv(
        env_profile=env_profile,
        config=config,
        model_name=model_name,
        evaluators=[BohnanzaEvaluator(max_turn_number=400)],
        terminal_evaluators=[],
        hide_unknown=True,
    )


def create_agents(
    agent_profiles: List[AgentProfile],
    env_profile: EnvironmentProfile,
    model_name: str | Dict[str, str] | List[str],
    config: Dict[str, Any],
) -> List[LLMAgent]:
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
    scenario = config.get("description", "Bohnanza").format(
        agent_names=", ".join(agent_names),
        max_rounds=config.get("max_rounds", 6),
    )
    env_profile = EnvironmentProfile(
        scenario=scenario,
        relationship=RelationshipType.acquaintance,
        agent_goals=agent_goals,
        tag="bohnanza",
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

    parser = argparse.ArgumentParser(description="Run a Bohnanza game.")
    parser.add_argument("--config", type=str, default=str(CONFIG_PATH))
    parser.add_argument("--roster", type=str, default=str(BASE_DIR / "roster.json"))
    args = parser.parse_args()

    config = load_config(args.config)
    roster = load_config(args.roster)
    config["agents"] = roster.get("agents", [])
    if "seed" in roster:
        config["seed"] = roster["seed"]

    agent_model_name = get_model_names(config)
    env, agents = prepare_scenario("gpt-4o", agent_model_name, config)

    print("Bohnanza")
    print("=" * 60)
    print_roster(config)
    print("=" * 60)

    await arun_one_episode(
        env=env,
        agent_list=agents,
        omniscient=False,
        script_like=False,
        json_in_script=False,
        tag="test_bohnanza",
        push_to_db=True,
    )


if __name__ == "__main__":
    LOG_FILE = BASE_DIR / "bohnanza_game_debug.log"
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
