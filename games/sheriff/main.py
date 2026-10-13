"""Launcher for the Sheriff of Nottingham game."""

from __future__ import annotations

import asyncio
import importlib.util
import os
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

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


# ============================================================================
# Evaluator
# ============================================================================


class SheriffEvaluator(SocialGameEndEvaluator):
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


def _shared(name: str) -> Any:
    """A helper module shared by the games, loaded by path like the game modules themselves."""
    key = f"games_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, BASE_DIR.parent / f"{name}.py")
        assert spec and spec.loader
        sys.modules[key] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sys.modules[key])
    return sys.modules[key]


# Only the move a reply commits to counts: a pack or inspection word that is negated, hypothetical, a question,
# about a later round or the subject of a sentence about it does not.
committed_move = _shared("committed_move")

# ---- Pack -------------------------------------------------------------------------------------------------------

# 'honest' (also "pack honestly"), 'smuggle' / 'smuggling' / 'contraband'. "Honestly, ..." and "To be honest, ..."
# are figures of speech, and "the gain from smuggling", "a history of smuggling" name the activity: not a pack.
_PACK_WORDS = {
    "honest": r"\bhonest\b|(?<=\bpack )honestly\b|(?<=\bpacking )honestly\b|(?<=\bplay )honestly\b",
    "smuggle": r"\bsmuggl(?:e|es)\b|(?<!\bfrom )(?<!\bof )(?<!\babout )(?<!\bfor )(?<!\bon )(?<!\bat )(?<!\bin )"
    r"\bsmuggling\b|\bcontraband\b",
}
_HONEST_IDIOMS = re.compile(
    r"\b(?:to be|if i'?m|if i am|if i'?m being) (?:totally |completely |fully |perfectly |brutally |quite |very )?"
    r"honest(?=\s*(?:,|with you))|\bhonest(?:ly)? speaking\b|\bbeing honest with you\b"
    r"|\bhonest (?:question|opinion|truth|answer|mistake)\b"
)
# What the merchant wants the Sheriff to believe is not its pack ("I'll convince Angela that I am honest"), nor is
# what it plans to say ("I'll confidently say, 'Laura, I've decided to pack honest goods'").
_PERSUASION = re.compile(
    r"\b(?:convinc\w*|persuad\w*|pretend\w*|appear\w*|seem\w*|claim\w*|tell\w*|told|say|said|bluff\w*|"
    r"make (?:him|her|them|the sheriff|everyone) (?:think|believe))\b"
)
_SPEECH = re.compile(r"\b(?:say|said|state|tell(?: \w+)?|told(?: \w+)?|claim|announce|reply|assure(?: \w+)?)\s*[,:]")


def _blank(m: "re.Match[str]") -> str:
    return " " * len(m.group())


def parse_pack(text: str, names: List[str] | None = None) -> Tuple[str | None, bool]:
    """Read a Pack reply as ('honest' | 'smuggle' | 'unclear' | None, whether it was the bare command).

    The reply asked for is exactly 'honest' or 'smuggle'. A longer reply is read by the choice it commits to
    (committed_move): "I'd rather smuggle than be honest" and "Honestly, I'll smuggle" are smuggle, "To be honest,
    smuggling pays more: smuggle" too. Not a choice: what the merchant means to tell the Sheriff ("I'll say I'm
    honest"), and another player's honesty ("Susan is honest"; `names` are the players). A reply that commits to
    both is 'unclear'. With no committed choice, a negated one counts as the other ("I am not going to smuggle" is
    honest); None if it names neither.
    """
    t = text.strip().lower().replace("’", "'")
    bare = re.sub(r"[^a-z ]", " ", t).split()
    if bare and bare[-1] in ("honest", "smuggle") and bare[:-1] in ([], ["pack"]):
        return bare[-1], True
    cleaned = _HONEST_IDIOMS.sub(_blank, committed_move.normalize(text))
    others = "".join(f"|{re.escape(n.lower())}" for n in names or [])
    third_person = re.compile(rf"\b(?:he|she|they|who|someone|everyone|nobody{others})\b")
    found = committed_move.mentions(cleaned, _PACK_WORDS, extra=_PERSUASION)
    chosen = []
    for m in found:
        head = re.split(r"[.!?;\n]", cleaned[: m.start])[-1]  # the sentence up to the mention
        clause = re.split(r",|\b(?:but|and|so|because|while)\b", head)[-1]
        if m.committed and not _SPEECH.search(head) and not third_person.search(clause):
            chosen.append(m.move)
    if len(set(chosen)) > 1:
        return "unclear", False
    if chosen:
        return chosen[0], False
    negated = {m.move for m in found if m.status == "negated"}
    if len(negated) == 1:
        return ("smuggle" if negated.pop() == "honest" else "honest"), False
    return None, False


# ---- Offers -----------------------------------------------------------------------------------------------------

_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20,
}
_NUM = r"(\d+|" + "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True)) + r")"
_AMOUNT = _NUM + r"\s*(?:gold(?:\s+coins?)?|coins?|g)\b"
# A bare number after the recipient ("I'll give you 2 if you pass me") is gold too: only gold changes hands here.
_BARE_AMOUNT = _NUM + r"(?=\s*(?:[.,;:!?)]|$|\b(?:if|to|for|and|so|now|this|right|as|upfront|up front)\b))"
_OFFER_VERBS = (
    r"offer|offers|offering|give|giving|gives|pay|paying|pays|hand|handing|hand over|slip|slipping|slide|sliding|"
    r"send|sending|promise|promising|bribe|bribing|tip|tipping|add|adding|throw in|transfer|transferring"
)
# The noun forms: "my offer is 3 gold", "a bribe of 2 gold", "a small token of appreciation, say 2 gold", "a small
# contribution of 2 gold", "a bribe, just 1 gold", "here's 2 gold", "a 2 gold token of appreciation".
_GIFT_NOUNS = r"offer|bribe|payment|tip|token|gift|sweetener|incentive|donation|contribution|present"
_OFFER_NOUN = re.compile(
    rf"\b(?:{_GIFT_NOUNS})\b[^.!?;\d]{{0,30}}?\b{_AMOUNT}"
    rf"|\bhere(?:'s| is)\s+{_AMOUNT}"
    rf"|{_AMOUNT}\s+(?:{_GIFT_NOUNS})\b"
)
_GAP = r"(?:\s+(?!\d)[\w'-]+){0,5}?\s+"
_SPEAKER = {"i", "i'm", "i'll", "i'd", "i've", "me", "my", "we", "we'll", "we'd", "let"}
_SOMEONE_ELSE = {
    "he", "she", "they", "someone", "everyone", "nobody", "who", "his", "her", "their", "you", "you'll", "you'd",
    "he'd", "she'd", "they'd", "he'll", "she'll", "they'll", "he's", "she's", "they're", "you're",
}
_OFFER_NEGATION = re.compile(
    r"\b(?:not|no|never|won'?t|don'?t|can'?t|cannot|without|refus\w*|instead of|rather than)\b|n't\b"
)
# Explicitly for a later round: "next round I'll give you 2", "I'll pay you 2 gold when I'm Sheriff".
_LATER_ROUND = re.compile(
    r"\b(?:next (?:round|time|turn)|later|future|another round|following round|someday|"
    r"when (?:i'?m|i am|you'?re|you are) (?:the )?(?:sheriff|a merchant|merchant))\b"
)
# An inspection's penalty, not a bribe: "if you inspect me, I pay 4 gold", "if he's honest, I'll pay 2 gold".
_INSPECTED = re.compile(
    r"\bif (?:you )?(?:inspect|catch|search|check|open)\w*|\bif (?:i'?m|i am|i get|i'?m being) (?:inspected|caught|searched)"
    r"|\bif (?:he|she|they)(?:'s| is| are|'re) (?:honest|smuggling)"
)
_NOW = re.compile(r"\b(?:now|this round|right away|immediately|up ?front)\b")
_PAST = re.compile(r"\b(?:last round|last time|earlier|already)\b")


def _someone_elses(clause: str, after: str, names: List[str]) -> bool:
    """The offer is reported, not made: its subject is another player or a pronoun ("Jonathan offering 1 gold",
    "Bob is offering 2", "you'll give me 2 gold"), or it is "the offer of 1 gold by Jonathan". "you" counts only
    right before the verb ("you give me"), not as an object ("straightforward for you and offer 10 gold")."""
    if re.match(r"\s*(?:by|from)\b", after):
        return True
    words = [w.strip("\"'()-,") for w in clause.split()[-4:]]
    for i, word in enumerate(reversed(words)):
        if word in _SPEAKER:
            return False
        if word == "you" and i > 0:
            continue
        if word in _SOMEONE_ELSE or word in names:
            return True
    return False


def _amount(word: str) -> int:
    return int(word) if word.isdigit() else _NUMBER_WORDS[word]


def _binds(t: str, m: "re.Match[str]", names: List[str], others: List[str]) -> bool:
    """Whether the offer matched by `m` in normalized text `t` binds (see parse_offer)."""
    sentence_start = max([0] + [b.end() for b in re.finditer(r"[.!?;\n]", t[: m.start()])])
    head = t[sentence_start: m.start()]
    clause = re.split(r",|\bbut\b|\bso\b", head)[-1]
    lead = re.split(r",|\bbut\b|\bso\b", head)[-2] if head.count(",") else ""
    _, right, _ = committed_move.contexts(t, m.start(), m.end())
    right = _NOW.split(right)[0]  # "2 gold now and more later": the 2 gold are for this round
    if _OFFER_NEGATION.search(clause):
        return False  # "I won't offer 2 gold", "I'm not paying 3 gold"
    if re.search(r"'s\s*$|\b(?:his|her|their|your)\s*$", head) or _someone_elses(clause, t[m.end():], names):
        return False  # "Charlie's offer of 3 gold", "Bob offers you 2 gold"
    if re.search(r"\b(?:me|us|myself|him|her|them)\b", t[m.start(): m.end()]):
        return False  # "give me 2 gold": the merchant would receive; "pay her 2 gold": not to the Sheriff
    if re.match(r"\s*(?:penalty|penalties|fine|fee|my way|to me|for me)\b", t[m.end():]) or _INSPECTED.search(head):
        return False  # "I'll pay the 2 gold penalty", "if you inspect me, I pay 4 gold": a penalty, not a bribe
    if others:
        alts = "|".join(re.escape(n) for n in others)
        if re.search(rf"\b(?:{alts})\b", t[m.start(): m.end()]) or re.match(rf"\s*to\s+(?:{alts})\b", t[m.end():]):
            return False  # "I'll give Bob 2 gold": not to the Sheriff
    span = t[m.start(): m.end()]
    if any(_LATER_ROUND.search(x) for x in (clause, span, right)) or (len(lead.split()) <= 4 and _LATER_ROUND.search(lead)):
        return False  # "next round I'll give you 2", "Next round, I'll pay 2 gold"
    if _PAST.search(clause) or _PAST.search(right):
        return False  # a payment already made
    return True


def parse_offer(text: str, names: List[str] | None = None, sheriff: str | None = None) -> int:
    """The gold a merchant's Negotiate statement binds itself to pay if passed (0 = no binding offer).

    A first-person offer to give or pay the Sheriff N gold in this round binds, in any wording: "I offer you 3
    gold", "I'll give you 2 gold", "I'd like to offer 2 gold", "I would offer you 2 gold", "my offer is 3 gold",
    "I'll give you 2 if you pass me". It does not bind when it is negated ("I won't pay you 2 gold"), made by
    someone else ("Bob offers you 2 gold", "Charlie's offer of 3 gold"), addressed to another player ("I'll give
    Bob 2 gold"), or explicitly for a later round ("next round I'll give you 2 gold"). If the statement makes
    several binding offers, the last one counts. `names` are the players, `sheriff` this round's Sheriff.
    """
    t = committed_move.normalize(text)
    lower_names = [n.lower() for n in names or []]
    others = [n for n in lower_names if n != (sheriff or "").lower()]
    to_sheriff = "you|the sheriff" + (f"|{re.escape(sheriff.lower())}" if sheriff else "")
    forms = [
        re.compile(rf"\b(?:{_OFFER_VERBS})\b{_GAP}{_AMOUNT}"),
        re.compile(rf"\b(?:{_OFFER_VERBS})\s+(?:to\s+)?(?:{to_sheriff})\s+{_BARE_AMOUNT}"),
        _OFFER_NOUN,
    ]
    found: List[Tuple[int, int]] = []
    for pattern in forms:
        for m in pattern.finditer(t):
            if _binds(t, m, lower_names, others):
                found.append((m.end(), _amount(next(g for g in m.groups() if g))))
    return max(found)[1] if found else 0


# ---- Inspect ----------------------------------------------------------------------------------------------------

_ALL_MERCHANTS = r"(?:all(?: of)?(?: the| you| them| three)?(?: merchants| bags)?|everyone|everybody|every merchant|each of you|all three)"


def parse_inspections(text: str, merchants: List[str]) -> Tuple[Dict[str, str], List[str]]:
    """The Sheriff's decisions: {merchant: 'inspect' | 'pass'}, and merchants given both decisions.

    A verb the reply commits to applies to the list of names right after it ("I will inspect Bob, Charlie and
    Diana", "Inspect: Bob, Charlie. Pass: Diana", "I'm inspecting Bob and passing Diana"), or to every merchant
    ("inspect all merchants"); "Bob: inspect" names the merchant first. A verb that does not commit ("I won't
    inspect Bob", "I could inspect Bob") decides nothing. A merchant named under both verbs gets no decision (and
    is passed with a notice).
    """
    if not merchants:
        return {}, []
    names = "|".join(re.escape(m.lower()) for m in sorted(merchants, key=len, reverse=True))
    name = rf"(?:{names})\b(?:'s\s+bag)?"
    glue = r"(?:\s*(?:,|&|\+|\band\b|\bthen\b)\s*)+(?:and\s+)?"
    lead = r"(?:\s*:)?(?:\s+(?:the|bags?|of|merchants?|on|through)\b)*\s*"
    verb = r"(inspect|inspects|inspecting|pass|passes|passing)"
    patterns = {
        "list": rf"\b{verb}\b{lead}({name}(?:{glue}{name})*|{_ALL_MERCHANTS}\b)",
        "named first": rf"\b({names})\b\s*[:\-–]\s*{verb}\b",
    }
    found: Dict[str, Set[str]] = {}
    for mention in committed_move.mentions(text, patterns):
        if not mention.committed:
            continue
        if mention.move == "list":
            word, listed = mention.match.group(1), mention.match.group(2)
            chosen = [x for x in merchants if re.search(rf"\b{re.escape(x.lower())}\b", listed)]
            if not chosen and re.fullmatch(_ALL_MERCHANTS, listed.strip()):
                chosen = list(merchants)
        else:
            chosen = [x for x in merchants if x.lower() == mention.match.group(1)]
            word = mention.match.group(2)
        decision = "inspect" if word.startswith("inspect") else "pass"
        for merchant in chosen:
            found.setdefault(merchant, set()).add(decision)
    decisions = {n: next(iter(v)) for n, v in found.items() if len(v) == 1}
    conflicts = [n for n in merchants if len(found.get(n, ())) > 1]
    return decisions, conflicts


def sheriff_rewards(gold: Dict[str, int]) -> Dict[str, float]:
    """Top half by gold +1, bottom half -1 (the middle seat of an odd table 0). Players tied on gold share the
    mean of the places they span, so the rewards always sum to zero: three tied for first get +1/3 each and the
    fourth -1 (rank_split_rewards gave those three 0 and the fourth -1, a sum of -1)."""
    n = len(gold)
    k = n // 2
    values = tuple(1.0 if i < k else -1.0 if i >= n - k else 0.0 for i in range(n))
    return graded_rank_rewards(gold, values)


def merchants_in_order(env: SocialDeductionGame) -> List[str]:
    """Merchants clockwise from the Sheriff's left (the next seats after the Sheriff)."""
    agents = list(env.agents)
    sheriff_idx = env.internal_state.get("sheriff_idx", 0)
    return [agents[(sheriff_idx + k) % len(agents)] for k in range(1, len(agents))]


class SheriffActionHandler(ActionHandler):
    def handle_action(
        self, env: SocialDeductionGame, agent_name: str, action: AgentAction
    ) -> None:
        if not isinstance(env, SheriffEnv):
            return

        agents = list(env.agents)
        sheriff_idx = env.internal_state.get("sheriff_idx", 0)
        sheriff = agents[sheriff_idx]
        acting = bool(env.action_mask[agents.index(agent_name)]) if env.action_mask else True
        if not acting:
            return

        if env.current_state == "Pack":
            if agent_name == sheriff:
                return  # Sheriff doesn't pack
            text = action.argument if action.action_type in ["action", "speak"] else ""
            pack, exact = parse_pack(text, [n for n in agents if n != agent_name])
            if pack in (None, "unclear"):
                # A reply that names neither choice, or commits to both, packs legal goods (no risk), as a
                # non-move does in other games; the merchant is told privately (a public line would reveal the bag).
                env.internal_state["packs"][agent_name] = "honest"
                said = (
                    "commits to both 'honest' and 'smuggle'" if pack == "unclear" else "named neither 'honest' nor 'smuggle'"
                )
                env.recv_message(
                    "Environment",
                    SimpleMessage(message=f"[Private] Your Pack reply {said}, so you pack honest goods this round."),
                    receivers=[agent_name],
                )
                return
            env.internal_state["packs"][agent_name] = pack
            if not exact:
                env.recv_message(
                    "Environment",
                    SimpleMessage(
                        message=f"[Private] Your Pack reply was not exactly 'honest' or 'smuggle'; "
                        f"it is read as '{pack}'."
                    ),
                    receivers=[agent_name],
                )

        elif env.current_state == "Negotiate":
            if agent_name == sheriff:
                return
            # Every merchant is told privately how its statement was read; a binding offer is also announced
            spoke = action.action_type in ["action", "speak"]
            amount = parse_offer(action.argument, agents, sheriff) if spoke else 0
            if amount > 0:
                env.internal_state["offers"][agent_name] = amount
                env.recv_message(
                    "Environment",
                    SimpleMessage(
                        message=f"[Game] {agent_name} makes a binding offer: {amount} gold to {sheriff} "
                        f"if {sheriff} passes {agent_name}'s bag."
                    ),
                )
                note = (
                    f"Your statement is read as a binding offer of {amount} gold: if {sheriff} passes your "
                    f"bag, you pay {sheriff} {amount} gold (at most what you then hold)."
                )
            else:
                said = "Your statement makes" if spoke else "You made no statement, so you make"
                note = (
                    f"{said} no binding offer: you pay nothing if {sheriff} passes your bag. "
                    f"Only a first-person offer of N gold for this round binds."
                )
            env.recv_message("Environment", SimpleMessage(message=f"[Private] {note}"), receivers=[agent_name])

        elif env.current_state == "Inspect":
            if agent_name != sheriff or action.action_type not in ["action", "speak"]:
                return  # Only the Sheriff inspects; no reply means no decisions (all passed)
            decisions, conflicts = parse_inspections(action.argument, merchants_in_order(env))
            env.internal_state["inspections"].update(decisions)
            if conflicts:
                env.internal_state["conflicting_decisions"] = conflicts

    def get_action_instruction(self, env: SocialDeductionGame, agent_name: str) -> str:
        if not isinstance(env, SheriffEnv):
            return ""

        agents = list(env.agents)
        sheriff_idx = env.internal_state.get("sheriff_idx", 0)
        sheriff = agents[sheriff_idx]
        round_num = env.internal_state.get("round", 0)
        total_rounds = env._config.get("total_rounds", 4)
        gold = env.internal_state.get("gold", {})

        if env.current_state == "Pack":
            if agent_name == sheriff:
                return (
                    f"Round {round_num + 1}/{total_rounds}. You are the SHERIFF this round. "
                    f"Gold: {gold}. Wait for others to pack their goods."
                )
            return (
                f"Round {round_num + 1}/{total_rounds}. {sheriff} is the Sheriff. "
                f"Gold: {gold}. "
                f"Reply with exactly one word: 'honest' (safe: 3 gold, no risk) or 'smuggle' "
                f"(risky: 8 gold if not caught, but -4 gold penalty if inspected). "
                f"The Sheriff will decide whether to inspect you."
            )

        elif env.current_state == "Negotiate":
            if agent_name == sheriff:
                return (
                    f"Round {round_num + 1}/{total_rounds}. You are the SHERIFF. "
                    f"Gold: {gold}. "
                    f"You speak after the merchants ({', '.join(merchants_in_order(env))}). "
                    f"They may offer bribes or plead their case. A merchant's offer of N gold for this round is "
                    f"binding: if you pass that merchant, you receive the gold. "
                    f"Remember: inspecting an honest merchant costs YOU 2 gold."
                )
            return (
                f"Round {round_num + 1}/{total_rounds}. {sheriff} is the Sheriff. "
                f"Gold: {gold}. "
                f"Negotiate with the Sheriff. You can claim you're honest, "
                f"offer bribes, or try to convince them not to inspect you. "
                f"An offer you make to give or pay the Sheriff N gold this round binds, in any wording ('I offer "
                f"you 3 gold', 'I'll give you 2 gold', 'I would offer you 2 gold'): if the Sheriff passes you, N "
                f"gold goes from you to the Sheriff (paid after your goods, at most what you then hold); if you "
                f"are inspected you pay nothing. An offer does not bind if it is negated ('I won't pay you 2 "
                f"gold'), made by someone else, or for a later round ('next round I'll give you 2 gold'), and if "
                "you make several offers the last one counts. Other promises are not enforced. "
                "You will be told privately how your statement was read."
            )

        elif env.current_state == "Inspect":
            if agent_name == sheriff:
                offers = env.internal_state.get("offers", {})
                offer_text = (
                    "Binding offers (paid to you only if you pass that merchant): "
                    + ", ".join(f"{n} {offers[n]} gold" for n in merchants_in_order(env) if n in offers)
                    + ". "
                    if offers
                    else "No binding offers this round. "
                )
                return (
                    f"Round {round_num + 1}/{total_rounds}. You are the SHERIFF. "
                    f"Gold: {gold}. "
                    f"For each merchant, decide: 'inspect NAME' or 'pass NAME' (a list after one word, "
                    f"'inspect NAME, NAME', applies to each name). "
                    f"Include ALL merchants in one response. "
                    f"Merchants: {', '.join(merchants_in_order(env))}. "
                    f"A merchant you give no decision for, or both, is passed. "
                    f"{offer_text}"
                    f"If you inspect an honest merchant, YOU pay them 2 gold. "
                    f"If you inspect a smuggler, THEY pay you 4 gold."
                )
            return (
                f"Round {round_num + 1}/{total_rounds}. {sheriff} is deciding whether to inspect. "
                f"Gold: {gold}. Waiting for the Sheriff's decision."
            )
        return ""


# ============================================================================
# Environment
# ============================================================================


class SheriffEnv(SocialDeductionGame):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(action_handler=SheriffActionHandler(), **kwargs)

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
            "sheriff_idx": 0,
            "gold": {name: 0 for name in self.agents},
            "packs": {},
            "inspections": {},
            "offers": {},
            "game_over": False,
            "final_scores": {},
            "end_reason": "",
        }

        agents = list(self.agents)
        sheriff = agents[0]
        self.recv_message(
            "Environment",
            SimpleMessage(
                message=f"[Game] Round 1. {sheriff} is the Sheriff this round."
            ),
        )
        # The base reset built the first observations before the game state existed ("Gold: {}"): rebuild the
        # instructions and deliver the round-1 announcement now instead of one turn late.
        for name in self.agents:
            if name not in obs:
                continue
            if self.action_handler:
                obs[name].action_instruction = self.action_handler.get_action_instruction(self, name)
            buffered = self.agent_message_buffer[name]
            if buffered:
                obs[name].last_turn = "\n".join(buffered)
                buffered.clear()
        return obs

    def _update_action_mask(self) -> None:
        """Sheriff doesn't pack; non-Sheriff don't inspect.

        Negotiate goes clockwise from the Sheriff's left, as declarations do in the
        rulebook, and the Sheriff speaks last, after hearing every merchant. (The base
        round-robin restarts at seat 0 each round, so the round-1 Sheriff spoke first.)
        """
        super()._update_action_mask()
        agents = list(self.agents)
        sheriff_idx = self.internal_state.get("sheriff_idx", 0)
        sheriff = agents[sheriff_idx]

        if self.current_state == "Pack":
            self.action_mask[sheriff_idx] = False
        elif self.current_state == "Negotiate":
            order = merchants_in_order(self) + [sheriff]
            spoken = getattr(self, "_state_turn_count", {}).get("Negotiate", 0)
            speaker = order[spoken % len(order)]
            for idx, name in enumerate(agents):
                self.action_mask[idx] = name == speaker
        elif self.current_state == "Inspect":
            for idx, name in enumerate(agents):
                self.action_mask[idx] = name == sheriff

    def _check_eliminations(self) -> None:
        if not self._should_transition_state():
            return

        if self.current_state == "Inspect":
            agents = list(self.agents)
            sheriff_idx = self.internal_state.get("sheriff_idx", 0)
            sheriff = agents[sheriff_idx]
            packs = self.internal_state.get("packs", {})
            inspections = self.internal_state.get("inspections", {})
            gold = self.internal_state["gold"]
            honest_val = self._config.get("honest_value", 3)
            smuggle_val = self._config.get("smuggle_value", 8)
            inspect_honest_pen = self._config.get("inspect_honest_penalty", 2)
            inspect_smuggle_pen = self._config.get("inspect_smuggle_penalty", 4)

            offers = self.internal_state.get("offers", {})
            conflicts = self.internal_state.pop("conflicting_decisions", [])
            if conflicts:
                self.recv_message(
                    "Environment",
                    SimpleMessage(
                        message=f"[Game] {sheriff} both inspected and passed {', '.join(conflicts)}, "
                        f"so {'that bag is' if len(conflicts) == 1 else 'those bags are'} passed."
                    ),
                )
            undecided = [n for n in merchants_in_order(self) if n not in inspections and n not in conflicts]
            if undecided:
                self.recv_message(
                    "Environment",
                    SimpleMessage(
                        message=f"[Game] {sheriff} gave no decision for {', '.join(undecided)}, "
                        f"so {'that bag is' if len(undecided) == 1 else 'those bags are'} passed."
                    ),
                )

            results = []
            for name in agents:
                if name == sheriff:
                    continue
                pack = packs.get(name, "honest")
                decision = inspections.get(name, "pass")

                if decision == "inspect":
                    if pack == "honest":
                        # Sheriff pays penalty
                        gold[sheriff] -= inspect_honest_pen
                        gold[name] += honest_val + inspect_honest_pen
                        results.append(
                            f"{name}: packed honest, inspected. "
                            f"{name}+{honest_val + inspect_honest_pen}, "
                            f"{sheriff}-{inspect_honest_pen}"
                        )
                    else:
                        # Smuggler caught
                        gold[name] -= inspect_smuggle_pen
                        gold[sheriff] += inspect_smuggle_pen
                        results.append(
                            f"{name}: CAUGHT smuggling! "
                            f"{name}-{inspect_smuggle_pen}, "
                            f"{sheriff}+{inspect_smuggle_pen}"
                        )
                else:
                    # Passed
                    if pack == "honest":
                        gold[name] += honest_val
                        results.append(
                            f"{name}: packed honest, passed. {name}+{honest_val}"
                        )
                    else:
                        gold[name] += smuggle_val
                        results.append(
                            f"{name}: smuggled successfully! {name}+{smuggle_val}"
                        )
                    # A binding bribe is paid when the bag is passed, out of the gold the
                    # merchant holds once its goods are paid.
                    offer = offers.get(name, 0)
                    if offer > 0:
                        paid = min(offer, max(0, gold[name]))
                        gold[name] -= paid
                        gold[sheriff] += paid
                        short = f" (offered {offer}, could pay {paid})" if paid < offer else ""
                        results[-1] += (
                            f". Bribe paid as offered: {name}-{paid}, {sheriff}+{paid}{short}"
                        )

            round_num = self.internal_state["round"] + 1
            self.internal_state["round"] = round_num

            self.recv_message(
                "Environment",
                SimpleMessage(
                    message=f"[Round {round_num} Results] Sheriff: {sheriff}. "
                    + "; ".join(results)
                    + f". Gold: {gold}"
                ),
            )

            # Reset for next round
            self.internal_state["packs"] = {}
            self.internal_state["inspections"] = {}
            self.internal_state["offers"] = {}

            total_rounds = self._config.get("total_rounds", 4)
            if round_num >= total_rounds:
                # Game over -- rank by gold; equal gold totals share a fate.
                sorted_players = sorted(gold.items(), key=lambda x: x[1], reverse=True)
                final = sheriff_rewards(gold)

                reason = (
                    f"Game over after {total_rounds} rounds. "
                    f"Final gold: {gold}. "
                    f"{sorted_players[0][0]} wins with {sorted_players[0][1]} gold!"
                )
                self.internal_state["game_over"] = True
                self.internal_state["final_scores"] = final
                self.internal_state["end_reason"] = reason
                self.recv_message(
                    "Environment", SimpleMessage(message=f"[Game] {reason}")
                )
            else:
                # Rotate Sheriff
                self.internal_state["sheriff_idx"] = round_num % len(agents)
                new_sheriff = agents[self.internal_state["sheriff_idx"]]
                self.recv_message(
                    "Environment",
                    SimpleMessage(
                        message=f"[Game] Round {round_num + 1}. "
                        f"{new_sheriff} is the Sheriff this round."
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
) -> SheriffEnv:
    return SheriffEnv(
        env_profile=env_profile,
        config=config,
        model_name=model_name,
        evaluators=[SheriffEvaluator(max_turn_number=50)],
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
    scenario = config.get("description", "Sheriff of Nottingham").format(
        agent_names=", ".join(agent_names)
    )
    env_profile = EnvironmentProfile(
        scenario=scenario,
        relationship=RelationshipType.acquaintance,
        agent_goals=agent_goals,
        tag="sheriff",
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

    parser = argparse.ArgumentParser(description="Run Sheriff of Nottingham game.")
    parser.add_argument("--config", type=str, default=str(CONFIG_PATH))
    parser.add_argument("--roster", type=str, default=str(BASE_DIR / "roster.json"))
    args = parser.parse_args()

    config = load_config(args.config)
    roster = load_config(args.roster)
    config["agents"] = roster.get("agents", [])

    agent_model_name = get_model_names(config)
    env, agents = prepare_scenario("gpt-4o", agent_model_name, config)

    print("Sheriff of Nottingham")
    print("=" * 60)
    print_roster(config)
    print("=" * 60)

    await arun_one_episode(
        env=env,
        agent_list=agents,
        omniscient=False,
        script_like=False,
        json_in_script=False,
        tag="test_sheriff",
        push_to_db=True,
    )


if __name__ == "__main__":
    LOG_FILE = BASE_DIR / "sheriff_game_debug.log"
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
