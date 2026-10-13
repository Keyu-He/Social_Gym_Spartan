"""Launcher for the Coup social game scenario."""

from __future__ import annotations

import asyncio
import importlib.util
import os
import logging
import random
import re
import sys
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

# Action -> claimed role mapping
ACTION_CLAIMS = {
    "tax": "Duke",
    "steal": "Captain",
    "assassinate": "Assassin",
    "exchange": "Ambassador",
}

# Action -> which roles can block it
ACTION_BLOCKERS = {
    "foreign_aid": ["Duke"],
    "steal": ["Captain", "Ambassador"],
    "assassinate": ["Contessa"],
}


CARD_TYPES = ["Duke", "Assassin", "Captain", "Ambassador", "Contessa"]

# The game ends after this many turns (one turn is one declaration, a rejected one included; about 12 rounds of
# 4 players). Env steps are capped separately as a safety net: a turn takes at most 7 steps. Both are backstops
# against a game that never ends, so the models are not told about them.
TURN_LIMIT = 50
MAX_ENV_STEPS = 400


def may_block(declaration: Dict[str, Any], responder: str) -> bool:
    """Official counteractions: anyone may block foreign aid (claiming Duke), only the target may block a steal
    or an assassination, and tax and exchange have no counteraction."""
    action_type = declaration.get("action")
    if action_type not in ACTION_BLOCKERS:
        return False
    if action_type == "foreign_aid":
        return True
    return responder == declaration.get("target")


# ============================================================================
# Reply parsing
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


# Only the move a reply commits to counts: a move word that is negated, hypothetical, a question, about a later
# turn or the subject of a sentence about it does not ("I don't think it's worth challenging here, so I pass").
committed_move = _shared("committed_move")


def negated(text: str, start: int) -> bool:
    """True when the word at `start` does not count as a move: negated, hypothetical, about a later turn, and so
    on (committed_move)."""
    t = committed_move.normalize(text)
    m = re.compile(r"\S+").match(t, start)
    return committed_move.status_of(t, start, m.end() if m else start) != committed_move.COMMITTED


_ACTION_RE = re.compile(
    r"\b(income|foreign(?:[\s_-]*aid)?|tax(?:es)?|steal(?:s|ing)?|assassinat(?:e|es|ing|ion)"
    r"|exchang(?:e|es|ing)|coups?)\b"
)
_ACTION_PREFIXES = [
    ("income", "income"),
    ("foreign", "foreign_aid"),
    ("tax", "tax"),
    ("steal", "steal"),
    ("assassinat", "assassinate"),
    ("exchang", "exchange"),
    ("coup", "coup"),
]


def _action_name(word: str) -> str:
    return next(name for prefix, name in _ACTION_PREFIXES if word.startswith(prefix))


_COUNTER_RE = re.compile(r"\b(?:block|blocks|blocking|blocked|counter\w*|challenge|challenges|challenging)\b")


def counter_worded(text: str, start: int) -> bool:
    """The action word at `start` sits inside block or challenge wording, so it is not declared: a block, counter
    or challenge word the reply commits to before it in the same sentence, or "with Contessa" (claiming, using) in
    that sentence ("I reveal my claim to block Christopher's Assassinate with Contessa!", "I challenge Susan's claim
    of Assassinate!"). "Since nobody can block tax, tax" is not: that block is negated."""
    low = committed_move.normalize(text)
    breaks = [m.end() for m in re.finditer(r"[.;!?\n]", low[:start])]
    first = breaks[-1] if breaks else 0
    after = re.search(r"[.;!?\n]", low[start:])
    last = start + after.start() if after else len(low)
    for m in committed_move.mentions(text, {"counter": _COUNTER_RE}):
        if first <= m.start < start and m.committed:
            return True
    return re.search(r"\b(?:with|claiming|using|by|revealing)\s+(?:my\s+|the\s+|a\s+)?contessa\b", low[first:last]) is not None


def _action_mentions(text: str) -> List[Any]:
    return [m for m in committed_move.mentions(text, {"action": _ACTION_RE}) if m.committed]


def counter_worded_actions(text: str) -> List[str]:
    """Actions whose committed word sits inside block or challenge wording (see counter_worded)."""
    found: List[str] = []
    for m in _action_mentions(text):
        action = _action_name(m.match.group(1))
        if counter_worded(text, m.start) and action not in found:
            found.append(action)
    return found


def declared_actions(text: str) -> List[Tuple[str, int, int]]:
    """The distinct actions a declaration commits to, in order of first mention, as (action, start, end).

    The action word may stand anywhere in the reply ("I declare income, taking 1 coin." is income); mentions that
    do not commit ("I won't steal", "income now, coup Brian later": the coup is for a later turn) and mentions
    inside block or challenge wording do not count.
    """
    found: List[Tuple[str, int, int]] = []
    for m in _action_mentions(text):
        if counter_worded(text, m.start):
            continue
        action = _action_name(m.match.group(1))
        if action not in [a for a, _, _ in found]:
            found.append((action, m.match.start(), m.match.end()))
    return found


_RESPONSES = {
    "challenge": r"\bchalleng(?:e|es|ing)\b",
    "block": r"\bblock(?:s|ing)?\b",
    "pass": r"\bpass(?:es|ing)?\b",
}


def response_keyword(text: str) -> str | None:
    """'challenge', 'block' or 'pass': the answer the reply commits to ("pass, I won't challenge" is a pass,
    "Challenging would be too risky. Pass." is a pass); 'unclear' when it commits to two different answers; None
    if it names none."""
    return committed_move.commitment(text, _RESPONSES)[0]


_CARD_RE = re.compile(r"\b(" + "|".join(c.lower() for c in CARD_TYPES) + r")s?\b")
_LOSE_VERBS = r"lose|losing|reveal|revealing|give up|giving up|discard|discarding|sacrifice|drop|flip|turn over"
_KEEP_VERBS = r"keep|keeping|protect|protecting|save|saving|hold on to"
_CARD = r"\s*:?\s*(?:my\s+|the\s+|a\s+|an\s+|one\s+)?(" + "|".join(c.lower() for c in CARD_TYPES) + r")s?\b"


def card_to_lose(text: str, hand: List[str]) -> str | None:
    """The card a 'lose CARD' reply gives up, or None if the reply names none of the player's cards, or two
    different cards to lose.

    Read in order: a held card after a losing verb the reply commits to ("lose Duke", "I reveal my Captain"); the
    other card when the reply keeps one ("keep Duke", "I won't lose my Duke"); the only held card the reply names;
    a held card that opens the reply ("Captain, since the Duke is more useful").
    """
    held = {c.lower(): c for c in hand}
    found = committed_move.mentions(text, {"lose": rf"\b(?:{_LOSE_VERBS}){_CARD}", "keep": rf"\b(?:{_KEEP_VERBS}){_CARD}"})
    lost = {held[m.match.group(1)] for m in found if m.move == "lose" and m.committed and m.match.group(1) in held}
    if len(lost) == 1:
        return lost.pop()
    if lost:
        return None
    kept = {
        held[m.match.group(1)]
        for m in found
        if m.match.group(1) in held and (
            (m.move == "keep" and m.committed) or (m.move == "lose" and m.status == "negated")
        )
    }
    others = list(dict.fromkeys(c for c in hand if c not in kept))
    if kept and len(others) == 1:
        return others[0]
    named = [
        m for m in committed_move.mentions(text, {"card": _CARD_RE}) if m.committed and m.match.group(1) in held
    ]
    distinct = list(dict.fromkeys(held[m.match.group(1)] for m in named))
    if len(distinct) == 1:
        return distinct[0]
    if named and not committed_move.normalize(text)[: named[0].start].strip():
        return held[named[0].match.group(1)]
    return None


def claimed_roles(text: str) -> List[str]:
    """The roles a reply claims: the card names it commits to, without negated ones ("I don't have a Captain")
    and other players' cards ("your Captain", "Bob's Duke")."""
    t = committed_move.normalize(text)
    roles: List[str] = []
    for m in committed_move.mentions(text, {"card": _CARD_RE}):
        if not m.committed or re.search(r"(?:'s|\b(?:your|his|her|their))\s+$", t[max(0, m.start - 8): m.start]):
            continue
        role = m.match.group(1).capitalize()
        if role not in roles:
            roles.append(role)
    return roles


def block_claim(text: str, action: str) -> Tuple[List[str], List[str]]:
    """(roles the block claims, roles that cannot block `action`).

    A block naming an allowed blocker claims the allowed roles it names ('block Captain'); one naming no role
    claims every allowed role; one naming only roles that cannot block the action ("I block the steal with my
    Contessa") is an invalid block, and the second list says which roles it named. The claim is never corrected.
    """
    allowed = ACTION_BLOCKERS.get(action, [])
    named = claimed_roles(text)
    if any(r in allowed for r in named):
        return [r for r in allowed if r in named], []
    if named:
        return [], named
    return list(allowed), []


def invalid_block_notice(player: str, action: str, wrong: List[str]) -> str:
    """The public line for a block whose claimed role cannot block the action."""
    allowed = " or ".join(ACTION_BLOCKERS.get(action, []))
    return (
        f"[Game] {player}'s block is invalid: it claims {' and '.join(wrong)}, which cannot block {action} (only "
        f"{allowed} can). It counts as no block."
    )


def named_block_role(text: str, roles: List[str]) -> str | None:
    """The blocking role a block names ('block Captain'), if it names exactly one of the allowed roles."""
    named = [r for r in claimed_roles(text) if r in roles]
    return named[0] if len(named) == 1 else None


_KEEP_WORDS = r"\b(?:keep|keeping|keeps|retain|retaining|hold onto|hold on to|stay with|stick with|go with|choose|choosing|pick|picking|select|selecting)\b\s*:?"
_RETURN_WORDS = (
    r"\b(?:return|returning|returns|give back|giving back|put back|putting back|send back|sending back|"
    r"shuffle back|discard|discarding|get rid of|getting rid of|ditch|ditching|drop|dropping|release|releasing)\b\s*:?"
)
_SCOPE_END = re.compile(
    r"[.;!?\n(]|\b(?:but|while|because|since|so|instead|then|and (?:i|then)|leaving)\b|" + _KEEP_WORDS + "|" + _RETURN_WORDS
)
_COUNT_WORDS = {"both": 2, "two": 2, "2": 2}


def _cards_after(t: str, end: int) -> List[str]:
    """The cards named from `end` to the end of that command ("keep Duke and Contessa; return ...")."""
    stop = _SCOPE_END.search(t, end)
    scope = t[end: stop.start() if stop else len(t)]
    cards: List[str] = []
    for m in re.finditer(r"(?:\b(both|two|2)\s+(?:the\s+|my\s+)?)?" + _CARD_RE.pattern, scope):
        n = _COUNT_WORDS.get(m.group(1) or "", 1)
        cards += [m.group(2).capitalize()] * n
    return cards


def read_keep(text: str, pool: List[str], keep_count: int) -> List[str] | None:
    """The cards an exchange reply keeps, or None when it does not name exactly `keep_count` cards of `pool`.

    The cards after the keep word the reply commits to count ("Return Captain and Ambassador; keep Duke and
    Contessa" keeps Duke and Contessa, "I won't keep the Captain, keep Duke Contessa" too); quoted or bold names
    and plurals ("Dukes") are names. A reply that only says what it returns keeps the rest; one with no keep or
    return word is read as the list of cards it names ("Duke Contessa").
    """
    t = committed_move.normalize(text)
    found = committed_move.mentions(text, {"keep": _KEEP_WORDS, "return": _RETURN_WORDS})

    def lists(kind: str) -> List[List[str]]:
        return [cards for m in found if m.move == kind and m.committed for cards in [_cards_after(t, m.match.end())] if cards]

    kept_lists, returned_lists = lists("keep"), lists("return")
    if kept_lists:
        if any(sorted(k) != sorted(kept_lists[0]) for k in kept_lists):
            return None
        kept = kept_lists[0]
    elif returned_lists:
        if any(sorted(r) != sorted(returned_lists[0]) for r in returned_lists):
            return None
        kept = list(pool)
        for c in returned_lists[0]:
            if c not in kept:
                return None
            kept.remove(c)
    elif not found:
        kept = [m.group(1).capitalize() for m in _CARD_RE.finditer(t)]
    else:
        return None
    left = list(pool)
    for c in kept:
        if c not in left:
            return None
        left.remove(c)
    return kept if len(kept) == keep_count else None


# ============================================================================
# Evaluator
# ============================================================================


class CoupEvaluator(SocialGameEndEvaluator):
    """Ends the game when one player is left, or at the turn limit (TURN_LIMIT declarations, checked when a turn
    is over), where the players holding the most cards share the win; `max_turn_number` caps env steps."""

    def __call__(
        self, turn_number: int, messages: List[Tuple[str, Message]], **kwargs: Any
    ) -> List[Tuple[str, Tuple[Tuple[str, int | float | bool], str]]]:
        env = kwargs.get("env")
        if not env:
            done = turn_number >= self.max_turn_number
            return [("environment", (("terminated", done), "Max turns reached." if done else ""))]

        alive_players = [n for n, alive in env.agent_alive.items() if alive]
        turn_limit = int(env._config.get("turn_limit", TURN_LIMIT))
        at_limit = (
            env.internal_state.get("turns_taken", 0) >= turn_limit
            and env.current_state == "Declare"
        )
        if len(alive_players) > 1 and (at_limit or turn_number >= self.max_turn_number):
            reason = f"Turn limit reached ({turn_limit} turns)."
            response: List[
                Tuple[str, Tuple[Tuple[str, int | float | bool], str]]
            ] = [("environment", (("terminated", True), reason))]
            hands = env.internal_state.get("hands", {})
            agent_names = list(env.agents)
            best = max((len(hands.get(n, [])) for n in agent_names), default=0)
            winners = [n for n in agent_names if len(hands.get(n, [])) == best]
            losers = [n for n in agent_names if n not in winners]
            # Players holding the most cards share the win, the others share the loss; if everyone ties it is
            # a draw (the old rule paid +1/4 to each of four tied players, which is not zero-sum).
            win_score = 1.0 / len(winners) if losers else 0.0
            loser_score = -1.0 / len(losers) if losers else 0.0
            for idx, name in enumerate(agent_names):
                key = f"agent_{idx + 1}"
                if name in winners:
                    label = "Winner (most cards)" if losers else "Draw (all tied on cards)"
                    response.append((key, (("complete_rating", win_score), label)))
                else:
                    response.append((key, (("complete_rating", loser_score), "Lost")))
            return response

        if len(alive_players) <= 1:
            agent_names = list(env.agents)
            winner = alive_players[0] if alive_players else None
            reason = (
                f"{winner} wins! Last player standing."
                if winner
                else "No players remaining."
            )
            response = [("environment", (("terminated", True), reason))]
            losers = [n for n in agent_names if n != winner]
            loser_score = -1.0 / len(losers) if losers else 0.0
            for idx, name in enumerate(agent_names):
                key = f"agent_{idx + 1}"
                if name == winner:
                    response.append((key, (("complete_rating", 1.0), "Winner")))
                else:
                    response.append(
                        (key, (("complete_rating", loser_score), "Eliminated"))
                    )
            return response

        return [("environment", (("terminated", False), ""))]


# ============================================================================
# Action Handler
# ============================================================================


class CoupActionHandler(ActionHandler):
    """Handles Coup game actions: declarations, challenges, blocks."""

    def handle_action(
        self, env: SocialDeductionGame, agent_name: str, action: AgentAction
    ) -> None:
        if not isinstance(env, CoupEnv):
            return
        if action.action_type not in ["action", "speak"]:
            return

        if env.current_state == "Declare":
            self._handle_declare(env, agent_name, action)
        elif env.current_state == "Response":
            self._handle_response(env, agent_name, action)
        elif env.current_state == "BlockChallenge":
            self._handle_block_challenge(env, agent_name, action)
        elif env.current_state == "Lose":
            self._handle_lose(env, agent_name, action)
        elif env.current_state == "Choose":
            self._handle_choose(env, agent_name, action)
        elif env.current_state == "Block":
            self._handle_block_offer(env, agent_name, action)

    def _handle_declare(
        self, env: CoupEnv, agent_name: str, action: AgentAction
    ) -> None:
        arg = action.argument.strip().lower()
        coins = env.internal_state.get("coins", {})
        my_coins = coins.get(agent_name, 0)

        # A declaration the engine cannot carry out loses the turn, and the public line below says so
        # (the response phase is skipped: there is nothing to answer).
        actions = declared_actions(arg)

        # Mandatory coup at 10+ coins ("income now, coup later" is not a coup)
        if my_coins >= 10 and [a for a, _, _ in actions] != ["coup"]:
            self._reject(env, agent_name, f"{agent_name} has {my_coins} coins and must coup.")
            return

        # The action word may stand anywhere in the reply; a reply naming no action, or several, is rejected, and
        # so is one whose action word sits inside block or challenge wording (a response written as a declaration)
        countered = counter_worded_actions(arg)
        if countered:
            self._reject(
                env,
                agent_name,
                f"it is unclear: {', '.join(countered)} stands inside block or challenge wording, which is not a "
                f"declaration. Name exactly one action.",
            )
            return
        if not actions:
            self._reject(
                env,
                agent_name,
                "unrecognized action. Use: income, foreign_aid, tax, steal NAME, "
                "assassinate NAME, exchange, or coup NAME.",
            )
            return
        if len(actions) > 1:
            self._reject(
                env,
                agent_name,
                f"it names more than one action ({', '.join(a for a, _, _ in actions)}); name exactly one.",
            )
            return

        action_type, target = self._parse_declaration(arg, list(env.agents), agent_name) or (actions[0][0], None)

        # Validate costs
        if action_type == "assassinate" and my_coins < 3:
            self._reject(env, agent_name, f"assassinate costs 3 coins; {agent_name} has {my_coins}.")
            return
        if action_type == "coup" and my_coins < 7:
            self._reject(env, agent_name, f"coup costs 7 coins; {agent_name} has {my_coins}.")
            return

        # Targeted actions need a living player other than the declarer
        if action_type in ("steal", "assassinate", "coup"):
            if target is None:
                self._reject(env, agent_name, f"{action_type} needs a target; name a living player.")
                return
            if not env.agent_alive.get(target, False):
                self._reject(env, agent_name, f"{target} is already eliminated.")
                return
            if target == agent_name:
                self._reject(env, agent_name, "a player cannot target itself.")
                return

        # Pay costs upfront: a successful challenge returns them, a block does not
        if action_type == "assassinate":
            coins[agent_name] = my_coins - 3

        # Store the declaration
        env.internal_state["current_declaration"] = {
            "declarer": agent_name,
            "action": action_type,
            "target": target,
            "claimed_role": ACTION_CLAIMS.get(action_type),
        }
        env.internal_state["responses"] = {}

        # Handle unchallengeable/unblockable actions immediately
        if action_type == "income":
            coins[agent_name] = my_coins + 1
            env.recv_message(
                "Environment",
                SimpleMessage(
                    message=f"[Game] {agent_name} takes Income (+1 coin, now {coins[agent_name]})."
                ),
            )
            env.internal_state["current_declaration"] = None
            # Skip response phase by forcing state transition
            env.internal_state["skip_response"] = True
            return

        if action_type == "coup" and target is not None:  # a coup without a target was rejected above
            coins[agent_name] = my_coins - 7
            env.recv_message(
                "Environment",
                SimpleMessage(
                    message=f"[Game] {agent_name} launches a Coup against {target}! (paid 7 coins)"
                ),
            )
            # The target loses an influence, choosing the card, once the (empty) response step is over
            self._queue_loss(env, target, f"{agent_name} launched a Coup against you")
            env.internal_state["current_declaration"] = None
            env.internal_state["skip_response"] = True
            return

        # Announce the declaration
        claimed = ACTION_CLAIMS.get(action_type, "")
        claim_str = f" (claiming {claimed})" if claimed else ""
        target_str = f" targeting {target}" if target else ""
        env.recv_message(
            "Environment",
            SimpleMessage(
                message=f"[Game] {agent_name} declares {action_type}{target_str}{claim_str}."
            ),
        )

    def _handle_response(
        self, env: CoupEnv, agent_name: str, action: AgentAction
    ) -> None:
        declaration = env.internal_state.get("current_declaration")
        if not declaration:
            return

        arg = action.argument.strip().lower()
        if "responses" not in env.internal_state:
            env.internal_state["responses"] = {}

        # The answer the reply commits to counts (committed_move); a reply committing to two answers is unclear.
        # Only character claims can be challenged (foreign aid claims none), and only the official blockers may
        # block. A challenge or block the action does not allow counts as a pass, with a public notice; so does a
        # block claiming a role that cannot block this action (the claim is not corrected).
        keyword = response_keyword(arg)
        said_challenge = keyword == "challenge"
        said_block = keyword == "block"
        may_challenge = declaration.get("claimed_role") is not None
        allowed_block = may_block(declaration, agent_name)
        action_type = declaration.get("action", "")
        response = "pass"
        if said_challenge and may_challenge:
            response = "challenge"
        elif said_block and allowed_block:
            roles, wrong = block_claim(arg, action_type)
            if roles:
                response = "block"
                env.internal_state.setdefault("block_roles", {})[agent_name] = roles
            else:
                env.recv_message("Environment", SimpleMessage(message=invalid_block_notice(agent_name, action_type, wrong)))
        else:
            if said_challenge:
                why = f"{action_type} claims no character and cannot be challenged."
            elif said_block and action_type not in ACTION_BLOCKERS:
                why = f"{action_type} cannot be blocked."
            elif said_block:
                why = f"only {declaration.get('target')}, the target, may block a {action_type}."
            else:
                why = ""
            if why:
                word = "challenge" if said_challenge else "block"
                env.recv_message(
                    "Environment",
                    SimpleMessage(
                        message=f"[Game] {agent_name}'s {word} is not allowed: {why} It counts as a pass."
                    ),
                )
            elif keyword == "unclear":
                env.recv_message(
                    "Environment",
                    SimpleMessage(
                        message=f"[Game] {agent_name}'s answer is unclear (it commits to more than one of challenge, "
                        f"block and pass). It counts as a pass."
                    ),
                )
            elif keyword != "pass":
                # Neither a challenge, a block nor a pass (as in Skull): counted as a pass, and said so
                env.recv_message(
                    "Environment",
                    SimpleMessage(
                        message=f"[Game] {agent_name} neither challenged, blocked nor passed. It counts as a pass."
                    ),
                )
        env.internal_state["responses"][agent_name] = response

    def _handle_block_challenge(
        self, env: CoupEnv, agent_name: str, action: AgentAction
    ) -> None:
        """Any player but the blocker may challenge a block (official rules); anything else is a pass."""
        block = env.internal_state.get("pending_block")
        if not block or agent_name == block["blocker"]:
            return
        keyword = response_keyword(action.argument)
        if keyword == "challenge":
            response = "challenge"
        else:
            response = "pass"
            if keyword == "block":
                env.recv_message(
                    "Environment",
                    SimpleMessage(
                        message=f"[Game] {agent_name}'s block is not allowed: only a challenge or a pass can answer "
                        f"a block. It counts as a pass."
                    ),
                )
            elif keyword == "unclear":
                env.recv_message(
                    "Environment",
                    SimpleMessage(
                        message=f"[Game] {agent_name}'s answer is unclear (it commits to more than one answer). "
                        f"It counts as a pass."
                    ),
                )
            elif keyword != "pass":
                env.recv_message(
                    "Environment",
                    SimpleMessage(
                        message=f"[Game] {agent_name} neither challenged nor passed. It counts as a pass."
                    ),
                )
        env.internal_state.setdefault("block_responses", {})[agent_name] = response

    def _handle_lose(self, env: CoupEnv, agent_name: str, action: AgentAction) -> None:
        """The player losing an influence names the card it reveals; a reply naming none of its cards, or two
        different cards to lose, loses a card chosen by a seeded random draw, said publicly."""
        loss = env.internal_state.get("pending_loss")
        if not loss or agent_name != loss["player"]:
            return
        hand = env.internal_state["hands"].get(agent_name, [])
        card = card_to_lose(action.argument, hand)
        if card is None:
            card = env.break_tie(list(hand))
            held = {c.lower() for c in hand}
            named = any(m.match.group(1) in held for m in committed_move.mentions(action.argument, {"card": _CARD_RE}))
            what = (
                f"does not say which one of {agent_name}'s cards it loses" if named
                else f"names none of {agent_name}'s cards"
            )
            env.recv_message(
                "Environment",
                SimpleMessage(message=f"[Game] {agent_name}'s reply {what}, so the card lost is drawn at random."),
            )
        env.internal_state["pending_loss"] = None
        self._reveal(env, agent_name, card)

    def _handle_choose(
        self, env: CoupEnv, agent_name: str, action: AgentAction
    ) -> None:
        """The exchanging player names the cards it keeps (read_keep). A reply that does not name exactly the
        right number of the offered cards keeps the original hand, said publicly (the rules text says so)."""
        pending = env.internal_state.get("pending_exchange")
        if not pending or agent_name != pending.get("declarer"):
            return
        all_cards: List[str] = list(pending["all_cards"])
        keep_count: int = pending["keep_count"]
        kept = read_keep(action.argument, all_cards, keep_count)
        if kept is None:
            env.recv_message(
                "Environment",
                SimpleMessage(
                    message=f"[Game] {agent_name}'s exchange choice could not be read (it must name exactly "
                    f"{keep_count} of the offered cards), so {agent_name} keeps the original cards."
                ),
            )
            kept = all_cards[:keep_count]
        self._finish_exchange(env, agent_name, kept)

    def _finish_exchange(self, env: CoupEnv, agent_name: str, kept: List[str]) -> None:
        pending = env.internal_state.get("pending_exchange") or {}
        returned = list(pending.get("all_cards", []))
        for c in kept:
            returned.remove(c)
        hands = env.internal_state.get("hands", {})
        deck = env.internal_state.get("deck", [])
        hands[agent_name] = list(kept)
        deck.extend(returned)
        env.shuffle_deck(deck)
        env.recv_message(
            "Environment",
            SimpleMessage(message=f"[Private] You kept {kept}. Returned {returned} to the deck."),
            receivers=[agent_name],
        )
        env.recv_message(
            "Environment",
            SimpleMessage(message=f"[Game] {agent_name} completes the exchange."),
        )
        env.internal_state["pending_exchange"] = None

    def _reject(self, env: CoupEnv, agent_name: str, reason: str) -> None:
        """End the turn of a declaration the engine cannot carry out, with a public line saying why."""
        env.recv_message(
            "Environment",
            SimpleMessage(
                message=f"[Game] {agent_name}'s declaration is rejected: {reason} {agent_name} loses the turn."
            ),
        )
        env.internal_state["current_declaration"] = None
        env.internal_state["skip_response"] = True

    @staticmethod
    def _find_named_player(words: List[str], names: List[str]) -> str | None:
        """The seated player named first among `words`, ignoring case and punctuation ("from Bob." -> Bob)."""
        text = " " + re.sub(r"[^\w]+", " ", " ".join(words)).lower() + " "
        best: Tuple[int, str] | None = None
        for name in names:
            key = " " + re.sub(r"[^\w]+", " ", name).lower().strip() + " "
            pos = text.find(key)
            if pos >= 0 and (best is None or pos < best[0]):
                best = (pos, name)
        return best[1] if best else None

    def _parse_declaration(
        self, arg: str, names: List[str], declarer: str | None = None
    ) -> Tuple[str, str | None] | None:
        """Parse a declaration into (action_type, target); None unless it names exactly one action.

        The target is the first player named after the action word ("I steal from Bob."), else the first other
        player named before it ("Bob looks rich, so I steal").
        """
        actions = declared_actions(arg)
        if len(actions) != 1 or counter_worded_actions(arg):
            return None
        action_type, start, end = actions[0]
        target = None
        if action_type in ["steal", "assassinate", "coup"]:
            target = self._find_named_player(arg[end:].split(), names) or self._find_named_player(
                arg[:start].split(), [n for n in names if n != declarer]
            )
        return (action_type, target)

    def _queue_loss(self, env: CoupEnv, player: str, reason: str) -> None:
        """`player` must lose an influence; the player chooses the card (Lose step) when the turn resolves."""
        env.internal_state.setdefault("loss_queue", []).append({"player": player, "reason": reason})

    def _reveal(self, env: CoupEnv, player: str, card: str) -> None:
        """`player` reveals `card` and loses it (official rules: the player chooses which card)."""
        hands = env.internal_state.get("hands", {})
        hand = hands.get(player, [])
        if card in hand:
            hand.remove(card)
            env.internal_state.setdefault("revealed", []).append(card)
            env.recv_message(
                "Environment",
                SimpleMessage(message=f"[Game] {player} reveals and loses: {card}."),
            )
            if hand:
                env.recv_message(
                    "Environment",
                    SimpleMessage(message=f"[Private] Your remaining card: {hand}"),
                    receivers=[player],
                )
        if not hand and env.agent_alive.get(player, False):
            env.agent_alive[player] = False
            env.recv_message(
                "Environment",
                SimpleMessage(
                    message=f"[Game] {player} has lost all influence and is eliminated!"
                ),
            )

    def _swap_revealed_card(self, env: CoupEnv, player: str, role: str) -> None:
        """A player who proves a claim shuffles the card back into the deck and draws a replacement."""
        hand = env.internal_state["hands"].get(player, [])
        deck = env.internal_state.get("deck", [])
        hand.remove(role)
        deck.append(role)
        env.shuffle_deck(deck)
        if deck:
            new_card = deck.pop()
            hand.append(new_card)
            env.recv_message(
                "Environment",
                SimpleMessage(
                    message=f"[Private] You shuffled {role} back and drew: {new_card}. Your hand: {hand}"
                ),
                receivers=[player],
            )

    def _resolve_responses(self, env: CoupEnv) -> None:
        """Resolve the answers to a declaration, as in the rulebook: a challenge of the claim first; if it fails
        (or nobody challenges), a block given in the same step stands and can itself be challenged
        (BlockChallenge step); otherwise the action takes effect. Sets internal_state["stage"]."""
        st = env.internal_state
        declaration = st.get("current_declaration")
        if not declaration:
            st["stage"] = "end"
            return

        responses = st.get("responses", {})
        declarer = declaration["declarer"]
        action_type = declaration["action"]
        claimed_role = declaration.get("claimed_role")
        coins = st.get("coins", {})
        hands = st.get("hands", {})

        # Only a character claim can be challenged, and only the official blockers can block
        # (_handle_response already turns anything else into a pass). When several players challenge (or block)
        # in the same step, the first one clockwise from the declarer does.
        challengers = env.clockwise_from(
            declarer, [n for n, r in responses.items() if r == "challenge"] if claimed_role else []
        )
        blockers = env.clockwise_from(
            declarer, [n for n, r in responses.items() if r == "block" and may_block(declaration, n)]
        )

        if challengers:
            challenger = challengers[0]
            declarer_hand = hands.get(declarer, [])

            if claimed_role and claimed_role in declarer_hand:
                # Declarer has the card - challenger loses
                env.recv_message(
                    "Environment",
                    SimpleMessage(
                        message=f"[Game] {challenger} challenges {declarer}'s claim of {claimed_role}. "
                        f"{declarer} reveals {claimed_role} - challenge FAILS! "
                        f"{challenger} loses influence."
                    ),
                )
                self._queue_loss(env, challenger, f"your challenge of {declarer}'s {claimed_role} failed")
                self._swap_revealed_card(env, declarer, claimed_role)
                # The challenge is over; a block given in the same step still stands (rulebook order)
                blockers = [b for b in blockers if b != challenger]
                self._set_block_or_apply(env, blockers)
                # A target that challenged a steal or an assassination answered before the challenge was settled;
                # as in the rulebook, it may still block once its challenge has failed (Block step)
                target = declaration.get("target")
                if st["stage"] == "apply" and action_type in ("steal", "assassinate") and responses.get(target) == "challenge":
                    st["stage"] = "offer_block"
            else:
                # Declarer doesn't have it - declarer loses, the action is cancelled and its cost returned
                env.recv_message(
                    "Environment",
                    SimpleMessage(
                        message=f"[Game] {challenger} challenges {declarer}'s claim of {claimed_role}. "
                        f"{declarer} does NOT have {claimed_role} - challenge SUCCEEDS! "
                        f"{declarer} loses influence."
                    ),
                )
                self._queue_loss(env, declarer, f"{challenger} challenged your {claimed_role} and you do not have it")
                if action_type == "assassinate":
                    coins[declarer] = coins.get(declarer, 0) + 3
                st["stage"] = "end"
        else:
            self._set_block_or_apply(env, blockers)
        st["responses"] = {}

    def _set_block_or_apply(self, env: CoupEnv, blockers: List[str]) -> None:
        st = env.internal_state
        if blockers:
            blocker = blockers[0]
            roles = st.get("block_roles", {}).get(blocker) or ACTION_BLOCKERS[st["current_declaration"]["action"]]
            st["pending_block"] = {"blocker": blocker, "roles": list(roles)}
            st["stage"] = "block"
        else:
            st["stage"] = "apply"

    def _resolve_block_challenge(self, env: CoupEnv) -> None:
        """Resolve the answers to a block: a successful challenge breaks the block (the blocker loses an influence
        and the action goes ahead); a failed one costs the challenger an influence and the block stands."""
        st = env.internal_state
        declaration = st.get("current_declaration") or {}
        block = st.get("pending_block") or {}
        blocker = block.get("blocker", "")
        roles: List[str] = block.get("roles", [])
        declarer = declaration.get("declarer", "")
        action_type = declaration.get("action", "")
        spent = " The 3 coins stay spent." if action_type == "assassinate" else ""
        # Several challengers: the first one clockwise from the declarer (the declarer itself first)
        challengers = env.clockwise_from(
            declarer, [n for n, r in st.get("block_responses", {}).items() if r == "challenge"], include_start=True
        )
        claim = " or ".join(roles)
        if not challengers:
            env.recv_message(
                "Environment",
                SimpleMessage(
                    message=f"[Game] Nobody challenges {blocker}'s block. {declarer}'s {action_type} is cancelled.{spent}"
                ),
            )
            st["stage"] = "end"
        else:
            challenger = challengers[0]
            held = next((r for r in roles if r in st["hands"].get(blocker, [])), None)
            if held:
                env.recv_message(
                    "Environment",
                    SimpleMessage(
                        message=f"[Game] {challenger} challenges {blocker}'s block. {blocker} reveals {held} - "
                        f"challenge FAILS! {challenger} loses influence. The block stands: {declarer}'s "
                        f"{action_type} is cancelled.{spent}"
                    ),
                )
                self._queue_loss(env, challenger, f"your challenge of {blocker}'s block failed")
                self._swap_revealed_card(env, blocker, held)
                st["stage"] = "end"
            else:
                env.recv_message(
                    "Environment",
                    SimpleMessage(
                        message=f"[Game] {challenger} challenges {blocker}'s block. {blocker} does NOT have "
                        f"{claim} - challenge SUCCEEDS! {blocker} loses influence, and {declarer}'s "
                        f"{action_type} goes ahead."
                    ),
                )
                self._queue_loss(env, blocker, f"{challenger} challenged your block and you do not have {claim}")
                st["stage"] = "apply"
        st["pending_block"] = None
        st["block_responses"] = {}

    def _handle_block_offer(self, env: CoupEnv, agent_name: str, action: AgentAction) -> None:
        """The target whose challenge failed answers 'block' or 'pass' (anything else is a pass, said publicly)."""
        if agent_name != env.internal_state.get("block_offer"):
            return
        env.internal_state["block_offer_reply"] = action.argument

    def _resolve_block_offer(self, env: CoupEnv) -> None:
        st = env.internal_state
        target = st.pop("block_offer", None)
        reply = st.pop("block_offer_reply", None)
        declaration = st.get("current_declaration") or {}
        action_type = declaration.get("action", "")
        st["stage"] = "apply"
        if reply is None:
            env.recv_message("Environment", SimpleMessage(message=f"[Game] {target} gave no response. It counts as a pass."))
            return
        keyword = response_keyword(reply)
        if keyword == "block":
            roles, wrong = block_claim(reply, action_type)
            if roles:
                st["pending_block"] = {"blocker": target, "roles": roles}
                st["stage"] = "block"
            else:
                env.recv_message("Environment", SimpleMessage(message=invalid_block_notice(target, action_type, wrong)))
        elif keyword == "challenge":
            env.recv_message(
                "Environment",
                SimpleMessage(message=f"[Game] {target}'s challenge is not allowed: the claim was already challenged. "
                              f"It counts as a pass."),
            )
        elif keyword == "unclear":
            env.recv_message(
                "Environment",
                SimpleMessage(message=f"[Game] {target}'s answer is unclear (it commits to both block and pass). "
                              f"It counts as a pass."),
            )
        elif keyword != "pass":
            env.recv_message("Environment", SimpleMessage(message=f"[Game] {target} neither blocked nor passed. It counts as a pass."))

    def _apply_action(self, env: CoupEnv, declaration: Dict[str, Any]) -> None:
        """Apply the effect of a successful action."""
        action_type = declaration["action"]
        declarer = declaration["declarer"]
        target = declaration.get("target")
        coins = env.internal_state.get("coins", {})
        hands = env.internal_state.get("hands", {})
        deck = env.internal_state.get("deck", [])

        if action_type == "foreign_aid":
            coins[declarer] = coins.get(declarer, 0) + 2
            env.recv_message(
                "Environment",
                SimpleMessage(
                    message=f"[Game] {declarer} takes Foreign Aid (+2 coins, now {coins[declarer]})."
                ),
            )
        elif action_type == "tax":
            coins[declarer] = coins.get(declarer, 0) + 3
            env.recv_message(
                "Environment",
                SimpleMessage(
                    message=f"[Game] {declarer} collects Tax (+3 coins, now {coins[declarer]})."
                ),
            )
        elif action_type == "steal":
            if target:
                stolen = min(2, coins.get(target, 0))
                coins[target] = coins.get(target, 0) - stolen
                coins[declarer] = coins.get(declarer, 0) + stolen
                env.recv_message(
                    "Environment",
                    SimpleMessage(
                        message=f"[Game] {declarer} steals {stolen} coins from {target}. "
                        f"({declarer}: {coins[declarer]}, {target}: {coins[target]})"
                    ),
                )
        elif action_type == "assassinate":
            if target and not env.agent_alive.get(target, False):
                # The target lost its last card earlier in this turn (a failed challenge or block)
                env.recv_message(
                    "Environment",
                    SimpleMessage(message=f"[Game] {target} is already out; the assassination has no effect."),
                )
            elif target:
                # Coins already paid upfront in _handle_declare
                env.recv_message(
                    "Environment",
                    SimpleMessage(
                        message=f"[Game] {declarer} assassinates {target}! (paid 3 coins)"
                    ),
                )
                self._queue_loss(env, target, f"{declarer} assassinated you")
        elif action_type == "exchange":
            # Draw 2 from deck. The declarer then picks which keep_count of the
            # combined (hand + drawn) pool to keep -- handled in Choose state.
            declarer_hand = list(hands.get(declarer, []))
            drawn = []
            for _ in range(min(2, len(deck))):
                drawn.append(deck.pop())

            all_cards = declarer_hand + drawn
            keep_count = len(declarer_hand)
            if keep_count > len(all_cards):
                keep_count = len(all_cards)

            env.internal_state["pending_exchange"] = {
                "declarer": declarer,
                "all_cards": all_cards,
                "keep_count": keep_count,
            }

            env.recv_message(
                "Environment",
                SimpleMessage(
                    message=(
                        f"[Private] Exchange: court deck dealt {drawn}. "
                        f"You now choose from {all_cards} (keep exactly {keep_count}). "
                        f"Reply with 'keep CARD' (1 influence) or 'keep CARD1 CARD2' (2 influence). "
                        f"Cards must be from: {sorted(set(all_cards))}. A reply that does not name exactly "
                        f"{keep_count} of them keeps your original cards."
                    )
                ),
                receivers=[declarer],
            )
            env.recv_message(
                "Environment",
                SimpleMessage(
                    message=f"[Game] {declarer} exchanges with the court deck (choosing which cards to keep)."
                ),
            )

    def get_action_instruction(self, env: SocialDeductionGame, agent_name: str) -> str:
        if not isinstance(env, CoupEnv):
            return ""

        coins = env.internal_state.get("coins", {})
        hands = env.internal_state.get("hands", {})
        revealed = env.internal_state.get("revealed", [])
        my_coins = coins.get(agent_name, 0)
        my_hand = hands.get(agent_name, [])

        # Player info
        player_info = []
        for name in env.agents:
            if env.agent_alive.get(name, False):
                card_count = len(hands.get(name, []))
                player_info.append(
                    f"{name}: {coins.get(name, 0)} coins, {card_count} card(s)"
                )
        players_str = "; ".join(player_info)

        revealed_str = f" Revealed cards: {revealed}." if revealed else ""

        if env.current_state == "Declare":
            forced = " You MUST coup (10+ coins)." if my_coins >= 10 else ""
            return (
                f"Your turn. Your cards: {my_hand}. Your coins: {my_coins}.{forced} "
                f"Players: [{players_str}].{revealed_str} "
                f"Actions: income, foreign_aid, tax (Duke), steal NAME (Captain), "
                f"assassinate NAME (Assassin, 3 coins), exchange (Ambassador), coup NAME (7 coins). "
                f"Name exactly one action."
            )
        elif env.current_state == "Response":
            declaration = env.internal_state.get("current_declaration", {})
            if declaration:
                declarer = declaration.get("declarer", "")
                action_type = declaration.get("action", "")
                target = declaration.get("target", "")
                claimed = declaration.get("claimed_role", "")

                target_str = f" targeting {target}" if target else ""
                claim_str = f" (claiming {claimed})" if claimed else ""

                # Offer only the legal responses: a challenge needs a character claim (not foreign aid);
                # a block needs a counteraction, and against steal or assassinate only the target has one.
                options = []
                if claimed:
                    options.append("'challenge' (risk a life to call bluff)")
                if may_block(declaration, agent_name):
                    roles = " or ".join(ACTION_BLOCKERS[action_type])
                    options.append(f"'block' (claim {roles})")
                note = (
                    f" Only {target}, the target, may block."
                    if action_type in ACTION_BLOCKERS and action_type != "foreign_aid"
                    and agent_name != target
                    else ""
                )
                if may_block(declaration, agent_name):
                    block_roles = ACTION_BLOCKERS[action_type]
                    which = (
                        f" You may name the role you claim ('block {block_roles[0]}' or 'block {block_roles[1]}'); "
                        f"a block naming neither claims both."
                        if len(block_roles) > 1
                        else ""
                    )
                    note += (
                        f" Any other player may challenge a block.{which} A block claiming any other role is invalid "
                        f"and counts as no block."
                    )
                    if claimed and action_type != "foreign_aid":
                        note += " If you challenge and your challenge fails, you may still block afterwards."
                return (
                    f"{declarer} declared {action_type}{target_str}{claim_str}. "
                    f"Your cards: {my_hand}. "
                    f"Choose: {', '.join(options)}, or 'pass'.{note} "
                    f"Anything other than a legal challenge or block counts as a pass. So does a reply that commits "
                    f"to two different answers."
                )
        elif env.current_state == "BlockChallenge":
            block = env.internal_state.get("pending_block") or {}
            declaration = env.internal_state.get("current_declaration") or {}
            blocker = block.get("blocker", "")
            claim = " or ".join(block.get("roles", []))
            if agent_name == blocker:
                return f"You blocked, claiming {claim}. Waiting to see whether anyone challenges your block."
            target = declaration.get("target")
            target_str = f" targeting {target}" if target else ""
            return (
                f"{blocker} blocks {declaration.get('declarer', '')}'s {declaration.get('action', '')}{target_str}, "
                f"claiming {claim}. Your cards: {my_hand}. Choose 'challenge' (you think {blocker} has no {claim}) "
                f"or 'pass'. If the challenge is right, {blocker} loses an influence and the action goes ahead; "
                f"if it is wrong, you lose an influence and the block stands. "
                f"Anything other than a challenge counts as a pass."
            )
        elif env.current_state == "Lose":
            loss = env.internal_state.get("pending_loss") or {}
            if loss.get("player") != agent_name:
                return f"Waiting for {loss.get('player', '')} to choose which card to reveal."
            choices = " or ".join(dict.fromkeys(my_hand))
            return (
                f"You must lose an influence: {loss.get('reason', '')}. Your cards: {my_hand}. "
                f"Reply 'lose CARD' naming the card you reveal and give up ({choices}). "
                f"A reply that names none of your cards, or two different cards to lose, loses one drawn at random."
            )
        elif env.current_state == "Choose":
            pending = env.internal_state.get("pending_exchange") or {}
            if pending.get("declarer") == agent_name:
                all_cards = pending.get("all_cards", [])
                keep_count = pending.get("keep_count", 0)
                return (
                    f"Exchange: choose from {all_cards}. Keep exactly {keep_count}. "
                    f"Reply with 'keep CARD' (1 influence) or 'keep CARD1 CARD2' (2 influence). "
                    f"A reply that does not name exactly {keep_count} of these cards keeps your original cards."
                )
        elif env.current_state == "Block":
            declaration = env.internal_state.get("current_declaration") or {}
            target = env.internal_state.get("block_offer")
            if agent_name != target:
                return f"Waiting for {target} to block or pass."
            roles = " or ".join(ACTION_BLOCKERS.get(declaration.get("action", ""), []))
            return (
                f"Your challenge of {declaration.get('declarer', '')}'s {declaration.get('action', '')} failed. "
                f"Your cards: {my_hand}. You may still block it: 'block' (claim {roles}) or 'pass'. A block "
                f"claiming any other role is invalid and counts as no block; anything else counts as a pass."
            )
        return ""


# ============================================================================
# Environment
# ============================================================================


class CoupEnv(SocialDeductionGame):
    """Coup bluffing game with challenges, blocks, and elimination."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(action_handler=CoupActionHandler(), **kwargs)

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

        card_types = self._config.get(
            "card_types", ["Duke", "Assassin", "Captain", "Ambassador", "Contessa"]
        )
        cards_per_type = self._config.get("cards_per_type", 3)
        starting_coins = self._config.get("starting_coins", 2)

        # Build deck. Every shuffle comes from the roster seed, so a roster replays the same deal and draws
        # (an unseeded config shuffles fresh); the random card lost by a player who names none comes from the
        # engine's seeded break_tie stream.
        seed = self._config.get("seed")
        self._deck_rng = random.Random(f"{seed}-deck") if seed is not None else random.Random()
        deck = card_types * cards_per_type
        self.shuffle_deck(deck)

        # Deal 2 cards to each player
        hands: Dict[str, List[str]] = {}
        coins: Dict[str, int] = {}
        for agent_name in self.agents:
            hands[agent_name] = [deck.pop(), deck.pop()]
            coins[agent_name] = starting_coins

        self.internal_state = {
            "hands": hands,
            "coins": coins,
            "deck": deck,
            "revealed": [],
            "current_declaration": None,
            "responses": {},
            "skip_response": False,
            "declarer_idx": 0,
            "pending_exchange": None,
            "turns_taken": 0,  # declarations so far, rejected ones included (the turn limit counts these)
            "stage": None,  # where the current turn's resolution stands: "block", "apply" or "end"
            "loss_queue": [],  # influence losses still to resolve this turn, in order
            "pending_loss": None,  # the loss whose owner is choosing a card (Lose step)
            "pending_block": None,  # the block awaiting challenges (BlockChallenge step)
            "block_roles": {},
            "block_responses": {},
            "next_state": None,
        }

        # Privately reveal cards to each player
        for agent_name in self.agents:
            self.recv_message(
                "Environment",
                SimpleMessage(
                    message=f"[Private] Your influence cards: {hands[agent_name]}. "
                    f"You have {starting_coins} coins."
                ),
                receivers=[agent_name],
            )

        self.recv_message(
            "Environment",
            SimpleMessage(
                message=f"[Game] Coup begins! Each player has 2 influence cards and {starting_coins} coins."
            ),
        )

        # The base reset built each player's first observation before the deal ("Your cards: []. Your
        # coins: 0"): rebuild the instruction, and deliver the deal now rather than one turn late, so the
        # opener declares knowing its cards.
        for agent_name in self.agents:
            if agent_name not in obs:
                continue
            if self.action_handler:
                obs[agent_name].action_instruction = (
                    self.action_handler.get_action_instruction(self, agent_name)
                )
            buffered = self.agent_message_buffer[agent_name]
            if buffered:
                obs[agent_name].last_turn = "\n".join(buffered)
                buffered.clear()

        return obs

    def shuffle_deck(self, deck: List[str]) -> None:
        if not hasattr(self, "_deck_rng"):
            self._deck_rng = random.Random()
        self._deck_rng.shuffle(deck)

    def _living_seat_from(self, seat: int, offset: int) -> int | None:
        """The first living seat at seat + offset, seat + offset + 1, ... in seat (clockwise) order."""
        agent_list = list(self.agents)
        n = len(agent_list)
        for k in range(n):
            idx = (seat + offset + k) % n
            if self.agent_alive.get(agent_list[idx], False):
                return idx
        return None

    def clockwise_from(self, start: str, names: List[str], include_start: bool = False) -> List[str]:
        """`names` in clockwise seat order from `start`: the players after it first (and `start` itself first of
        all if `include_start`)."""
        agent_list = list(self.agents)
        n = len(agent_list)
        seat = agent_list.index(start) if start in agent_list else 0
        offset = (lambda name: (agent_list.index(name) - seat) % n) if include_start else (
            lambda name: (agent_list.index(name) - seat - 1) % n
        )
        return sorted(names, key=offset)

    def _update_action_mask(self) -> None:
        """Override to handle declarer rotation and Response exclusion."""
        super()._update_action_mask()

        if self.current_state == "Declare":
            # Only the player whose turn it is acts. declarer_idx is that player's seat; turns go clockwise
            # among the living players (the seat is advanced when the turn ends, in _check_eliminations).
            agent_list = list(self.agents)
            if agent_list:
                seat = self._living_seat_from(
                    self.internal_state.get("declarer_idx", 0) % len(agent_list), 0
                )
                if seat is not None:
                    self.action_mask = [False] * len(agent_list)
                    self.action_mask[seat] = True

        elif self.current_state == "Response":
            declaration = self.internal_state.get("current_declaration")
            if declaration:
                declarer = declaration.get("declarer", "")
                agent_list = list(self.agents)
                # Exclude declarer from responding
                try:
                    idx = agent_list.index(declarer)
                    self.action_mask[idx] = False
                except ValueError:
                    pass
                # Exclude dead players
                for i, name in enumerate(agent_list):
                    if not self.agent_alive.get(name, False):
                        self.action_mask[i] = False
            # If skip_response (income/coup), no one acts
            if self.internal_state.get("skip_response", False):
                self.action_mask = [False] * len(self.agents)

        elif self.current_state == "BlockChallenge":
            # Every living player except the blocker may challenge the block
            block = self.internal_state.get("pending_block") or {}
            self.action_mask = [
                self.agent_alive.get(n, False) and n != block.get("blocker") for n in self.agents
            ]

        elif self.current_state == "Lose":
            loss = self.internal_state.get("pending_loss") or {}
            self.action_mask = [n == loss.get("player") for n in self.agents]

        elif self.current_state == "Block":
            target = self.internal_state.get("block_offer")
            self.action_mask = [n == target for n in self.agents]

        elif self.current_state == "Choose":
            # Only the declarer (with pending exchange) can act.
            pending = self.internal_state.get("pending_exchange") or {}
            declarer = pending.get("declarer", "")
            agent_list = list(self.agents)
            self.action_mask = [False] * len(agent_list)
            if declarer in agent_list:
                self.action_mask[agent_list.index(declarer)] = True

    # A turn: Declare, then Response (empty after income, a coup or a rejected declaration), then as needed a
    # Lose step for each influence lost with a choice of card, a BlockChallenge step after a block, and the Choose
    # step of an exchange. _check_eliminations decides the next state (internal_state["next_state"]) at the end of
    # every step; every state lasts one step.
    def _perform_transition_state(self) -> None:
        nxt = self.internal_state.pop("next_state", None)
        if not nxt:
            super()._perform_transition_state()
            return
        self.current_state = nxt
        if hasattr(self, "_state_turn_count"):
            self._state_turn_count[self.current_state] = 0
        if hasattr(self, "_round_robin_idx"):
            self._round_robin_idx = 0
        self.recv_message(
            "Environment",
            SimpleMessage(message=f"[Game] Entering state: {self.current_state}"),
        )

    def _run_resolution(self) -> None:
        """Carry the current turn forward until a player has to answer, or the turn is over.

        Influence losses come first, in order: a player with two different cards chooses which to reveal (Lose
        step); one card, or two of a kind, is lost without asking. Then a pending block is announced and opened to
        challenges, or the action takes effect (an assassination queues the target's loss; an exchange opens the
        Choose step). When nothing is left the turn passes to the next living seat clockwise.
        """
        st = self.internal_state
        handler = self.action_handler
        assert isinstance(handler, CoupActionHandler)
        while True:
            queue = st.setdefault("loss_queue", [])
            while queue:
                loss = queue.pop(0)
                hand = st["hands"].get(loss["player"], [])
                if not self.agent_alive.get(loss["player"], False) or not hand:
                    continue
                if len(set(hand)) == 1:
                    handler._reveal(self, loss["player"], hand[0])
                    continue
                st["pending_loss"] = loss
                st["next_state"] = "Lose"
                return
            stage = st.get("stage")
            if stage == "offer_block":
                target = st["current_declaration"].get("target")
                if self.agent_alive.get(target, False) and st["hands"].get(target):
                    declaration = st["current_declaration"]
                    roles = " or ".join(ACTION_BLOCKERS[declaration["action"]])
                    self.recv_message(
                        "Environment",
                        SimpleMessage(
                            message=f"[Game] {target}'s challenge failed, but {target} may still block "
                            f"{declaration['declarer']}'s {declaration['action']} (claiming {roles}) or pass."
                        ),
                    )
                    st["stage"] = None
                    st["block_offer"] = target
                    st["next_state"] = "Block"
                    return
                st["stage"] = "apply"
                continue
            if stage == "block":
                block = st["pending_block"]
                declaration = st["current_declaration"]
                if self.agent_alive.get(block["blocker"], False):
                    spent = " The 3 coins stay spent." if declaration["action"] == "assassinate" else ""
                    self.recv_message(
                        "Environment",
                        SimpleMessage(
                            message=f"[Game] {block['blocker']} blocks {declaration['declarer']}'s "
                            f"{declaration['action']}, claiming {' or '.join(block['roles'])}.{spent} "
                            f"Any other player may challenge the block."
                        ),
                    )
                    st["stage"] = None
                    st["block_responses"] = {}
                    st["next_state"] = "BlockChallenge"
                    return
                st["stage"] = "apply"
                continue
            if stage == "apply":
                st["stage"] = "end"
                handler._apply_action(self, st["current_declaration"])
                if st.get("pending_exchange"):
                    st["next_state"] = "Choose"
                    return
                continue
            # The turn is over: pass it to the next living seat clockwise from this turn's declarer
            st["stage"] = None
            st["current_declaration"] = None
            st["pending_block"] = None
            st["block_roles"] = {}
            st.pop("block_offer", None)
            agent_list = list(self.agents)
            seat = st.get("declarer_idx", 0) % len(agent_list)
            nxt = self._living_seat_from(seat, 1)
            st["declarer_idx"] = nxt if nxt is not None else seat
            st["next_state"] = "Declare"
            return

    def _fill_missing_answers(self, answers: Dict[str, str]) -> None:
        """A player who was asked but gave no action (a failed model call) passes, visibly."""
        for name, asked in zip(list(self.agents), self.action_mask):
            if asked and name not in answers:
                answers[name] = "pass"
                self.recv_message(
                    "Environment",
                    SimpleMessage(message=f"[Game] {name} gave no response. It counts as a pass."),
                )

    def _check_eliminations(self) -> None:
        """Resolve the step's answers and choose the next state (see _run_resolution), then check for dead
        players."""
        handler = self.action_handler
        agent_list = list(self.agents)
        st = self.internal_state

        # A Declare step that produced no declaration at all (e.g. a failed model call) loses the turn
        # visibly, like a rejected declaration, instead of opening a response round to nothing.
        if self.current_state == "Declare" and agent_list:
            seat = self._living_seat_from(st.get("declarer_idx", 0) % len(agent_list), 0)
            if seat is not None:
                st["declarer_idx"] = seat  # the seat that declared this turn
                st["turns_taken"] = st.get("turns_taken", 0) + 1
                if (
                    not st.get("current_declaration")
                    and not st.get("skip_response", False)
                    and isinstance(handler, CoupActionHandler)
                ):
                    handler._reject(self, agent_list[seat], "no action was given.")
            st["next_state"] = "Response"

        elif self.current_state == "Response" and isinstance(handler, CoupActionHandler):
            if st.get("skip_response", False):
                # Income/coup/rejected declaration: nothing to answer (a coup's target still loses a card)
                st["skip_response"] = False
                st["stage"] = "end"
            else:
                self._fill_missing_answers(st.setdefault("responses", {}))
                handler._resolve_responses(self)
            self._run_resolution()

        elif self.current_state == "BlockChallenge" and isinstance(handler, CoupActionHandler):
            self._fill_missing_answers(st.setdefault("block_responses", {}))
            handler._resolve_block_challenge(self)
            self._run_resolution()

        elif self.current_state == "Block" and isinstance(handler, CoupActionHandler):
            handler._resolve_block_offer(self)
            self._run_resolution()

        elif self.current_state == "Lose" and isinstance(handler, CoupActionHandler):
            loss = st.get("pending_loss")
            if loss:
                # No reply at all (a failed model call): a card drawn at random, said publicly
                hand = st["hands"].get(loss["player"], [])
                st["pending_loss"] = None
                if hand:
                    self.recv_message(
                        "Environment",
                        SimpleMessage(
                            message=f"[Game] {loss['player']} gave no choice of card, so the card lost is drawn at random."
                        ),
                    )
                    handler._reveal(self, loss["player"], self.break_tie(list(hand)))
            self._run_resolution()

        # No reply in Choose (a failed model call): the original cards are kept, as for an unreadable choice, said
        # publicly; the kept cards are named only to their owner.
        elif self.current_state == "Choose":
            pending = self.internal_state.get("pending_exchange")
            if pending and isinstance(handler, CoupActionHandler):
                declarer = pending["declarer"]
                self.recv_message(
                    "Environment",
                    SimpleMessage(
                        message=f"[Game] {declarer} did not specify a choice, so {declarer} keeps the original cards."
                    ),
                )
                handler._finish_exchange(self, declarer, list(pending["all_cards"])[: pending["keep_count"]])
            if isinstance(handler, CoupActionHandler):
                self._run_resolution()

        # Check for players with no cards
        hands = self.internal_state.get("hands", {})
        for agent_name in self.agents:
            if self.agent_alive.get(agent_name, False):
                if len(hands.get(agent_name, [])) == 0:
                    self.agent_alive[agent_name] = False


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
) -> CoupEnv:
    return CoupEnv(
        env_profile=env_profile,
        config=config,
        model_name=model_name,
        evaluators=[CoupEvaluator(max_turn_number=MAX_ENV_STEPS)],
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
    scenario = config.get("description", "Coup").format(
        agent_names=", ".join(agent_names)
    )
    env_profile = EnvironmentProfile(
        scenario=scenario,
        relationship=RelationshipType.acquaintance,
        agent_goals=agent_goals,
        tag="coup",
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

    parser = argparse.ArgumentParser(description="Run Coup game.")
    parser.add_argument("--config", type=str, default=str(CONFIG_PATH))
    parser.add_argument("--roster", type=str, default=str(BASE_DIR / "roster.json"))
    args = parser.parse_args()

    config = load_config(args.config)
    roster = load_config(args.roster)
    config["agents"] = roster.get("agents", [])

    agent_model_name = get_model_names(config)
    env, agents = prepare_scenario("gpt-4o", agent_model_name, config)

    print("Coup")
    print("=" * 60)
    print_roster(config)
    print("=" * 60)

    await arun_one_episode(
        env=env,
        agent_list=agents,
        omniscient=False,
        script_like=False,
        json_in_script=False,
        tag="test_coup",
        push_to_db=True,
    )


if __name__ == "__main__":
    LOG_FILE = BASE_DIR / "coup_game_debug.log"
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
