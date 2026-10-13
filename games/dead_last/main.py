"""Launcher for the Dead Last social game scenario."""

from __future__ import annotations

import asyncio
import copy
import os
import logging
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
# Parsing helpers
# ============================================================================

# A number not glued to a preceding digit, so "34-33-33" reads as 34, 33, 33 and " -10" as -10. A hyphen
# right after a letter is not a minus sign either, so "Carol-25" reads as 25.
_NUMBER_RE = re.compile(r"(?<![\d.])(?:(?<![A-Za-z])-)?\d+(?:\.\d+)?")

SEAT_ORDER = "in seat order"
BY_NAME = "by name"


def _name_re(name: str) -> str:
    return rf"(?<![A-Za-z]){re.escape(name)}(?![A-Za-z])"


def first_named(text: str, names: List[str]) -> str | None:
    """The name from `names` that appears first in `text` as a whole word (any case, punctuation allowed).

    The reader of points before 2026-10-05; kept for the audit scripts. Points are read by read_point."""
    found = []
    for name in names:
        m = re.search(_name_re(name), text, re.I)
        if m:
            found.append((m.start(), name))
    return min(found)[1] if found else None


# The pointer's own verb ("point Bob", "I'm pointing at Bob"); "points" and "pointed" usually report someone
# else's point ("Bob points at Carol") and are not anchors.
_POINT_WORD = re.compile(r"\b(?:point|pointing)\b", re.I)


def read_point(text: str, players: List[str]) -> str | None:
    """The player a point names, or None.

    The first of `players` (the whole roster, so the pointer and eliminated players are read too, and the caller
    rejects them) written as a whole word after the first 'point' or 'pointing'. A name written with its capital
    letter is tried before a lowercase one, so a common word ("mark") does not shadow a later name. A reply
    without 'point' or 'pointing', or with no player after it, names nobody. point_rule states this rule.
    """
    m = _POINT_WORD.search(text)
    if not m:
        return None
    rest = text[m.end():]
    for flags in (0, re.I):
        found = sorted((mm.start(), name) for name in players for mm in re.finditer(_name_re(name), rest, flags))
        if found:
            return found[0][1]
    return None


def point_rule(players: List[str]) -> str:
    """read_point in the words the Point instruction uses."""
    return (
        f"Your point names the first player of the game ({', '.join(players)}; this includes you and "
        f"eliminated players) written as a whole word after the first 'point' or 'pointing' in your reply; a "
        f"name written with its capital letter is tried before a lowercase one. A reply without 'point' or "
        f"'pointing', or with no player after it, names nobody."
    )


def _check_shares(tokens: List[str], survivors: List[str]) -> Tuple[Dict[str, int] | None, str]:
    """One whole, non-negative number per survivor (in `survivors` order), summing to 100."""
    n = len(survivors)
    if len(tokens) != n:
        return None, f"lists {len(tokens)} number(s), not {n}"
    if any("." in t for t in tokens):
        return None, "has a share that is not a whole number"
    values = [int(t) for t in tokens]
    if any(v < 0 for v in values):
        return None, "has a negative share"
    if sum(values) != 100:
        return None, f"sums to {sum(values)}, not 100"
    return dict(zip(survivors, values)), ""


_SENTENCE_END = re.compile(r"[.!?](?=\s|$)")


def _label_pairings(text: str, survivors: List[str], numbers: List[Tuple[int, str]],
                    mentions: List[Tuple[int, str]]) -> List[Tuple[str, ...]]:
    """Every distinct way to give each survivor one number written next to one of its mentions.

    In the offer's stream of survivor names and numbers (other words skipped), a number is next to a name when
    it comes right before or right after it, within the same sentence. Each number goes to at most one
    survivor. Returns the distinct share tuples (in `survivors` order) of all such pairings.
    """
    ends = [m.start() for m in _SENTENCE_END.finditer(text)]
    stream = sorted([(p, "name", n) for p, n in mentions] + [(p, "num", t) for p, t in numbers])
    options: Dict[str, set[int]] = {name: set() for name in survivors}
    for i, (pos, kind, name) in enumerate(stream):
        if kind != "name":
            continue
        for j in (i - 1, i + 1):
            if 0 <= j < len(stream) and stream[j][1] == "num":
                lo, hi = sorted((pos, stream[j][0]))
                if not any(lo <= e < hi for e in ends):
                    options[name].add(j)
    found: set[Tuple[str, ...]] = set()

    def assign(k: int, used: set[int], shares: List[str]) -> None:
        if k == len(survivors):
            found.add(tuple(shares))
            return
        for j in sorted(options[survivors[k]] - used):
            assign(k + 1, used | {j}, shares + [stream[j][2]])

    assign(0, set(), [])
    return sorted(found)


def read_offer(text: str, survivors: List[str]) -> Tuple[Dict[str, int] | None, str, str]:
    """A final offer, read by its labels when it names every survivor, otherwise in seat order.

    By label: each survivor gets the one number written next to its name, right before or right after it in
    the same sentence with no other name or number in between, in either direction and in any mix ("offer
    Carol 25, Alice 50, Bob 25", "50 for Alice, 25 for Bob, 25 for Carol", "Bob 25, Carol 25 and 50 for
    Alice"); each number goes to one survivor. Names match as whole words, in any case. If no such pairing
    exists, or several give different splits, the offer is unclear: it cannot be read, counts as a mismatch,
    and the result line says why. An offer that names only some survivors (or none) is read in seat order:
    exactly one whole number per survivor, in the listed order. Either way the shares must be whole,
    non-negative and sum to 100.

    Returns (split, "", how) or (None, why it cannot be read, how), with how SEAT_ORDER or BY_NAME.
    """
    numbers = [(m.start(), m.group()) for m in _NUMBER_RE.finditer(text)]
    mentions = [(m.start(), name) for name in survivors for m in re.finditer(_name_re(name), text, re.I)]
    if survivors and {name for _, name in mentions} == set(survivors):
        pairings = _label_pairings(text, survivors, numbers, mentions)
        if not pairings:
            return None, "names every survivor but does not write one number next to each name (read by name)", BY_NAME
        if len(pairings) > 1:
            return None, ("names every survivor, but its numbers can be paired with the names in more than one "
                          "way (read by name)"), BY_NAME
        split, problem = _check_shares(list(pairings[0]), survivors)
        return split, (f"{problem} (read by name)" if problem else ""), BY_NAME
    split, problem = _check_shares([t for _, t in numbers], survivors)
    return split, problem, SEAT_ORDER


# read_offer's by-label rule, in the words the Final_offer instruction and the rules text use.
LABEL_RULE = (
    "An offer that names every survivor is read only by its labels: each survivor gets the one number written "
    "right before or right after its name, in the same sentence, with no other name or number in between, and "
    "each number goes to one survivor (names match as whole words, in any case). If the numbers cannot be paired "
    "with the names that way, or can be paired in more than one way that gives different shares, the offer is unclear. An offer that names "
    "only some survivors is read in seat order."
)


def parse_offer(text: str, survivors: List[str]) -> Tuple[Dict[str, int] | None, str]:
    """(split, "") or (None, why it cannot be read); see read_offer."""
    split, problem, _ = read_offer(text, survivors)
    return split, problem


def format_split(split: Dict[str, int], survivors: List[str]) -> str:
    return ", ".join(f"{name} {split[name]}" for name in survivors)


# ============================================================================
# Evaluator
# ============================================================================


class DeadLastEvaluator(SocialGameEndEvaluator):
    """Evaluator that checks Dead Last win conditions."""

    def __call__(
        self, turn_number: int, messages: List[Tuple[str, Message]], **kwargs: Any
    ) -> List[Tuple[str, Tuple[Tuple[str, int | float | bool], str]]]:
        env = kwargs.get("env")
        # The turn cap only guards against a game that never ends (normal play ends long before it), so the
        # models are not told: players still in the game score 0 and eliminated players -1. A final decided on
        # the cap turn keeps its result: the offers are compared in one step, so the last turn can be the
        # Final_offer step.
        if turn_number >= self.max_turn_number and not (env and env.internal_state.get("game_over", False)):
            if env:
                response: List[
                    Tuple[str, Tuple[Tuple[str, int | float | bool], str]]
                ] = [("environment", (("terminated", True), "Max turns reached."))]
                agent_names = list(env.agents)
                for idx, name in enumerate(agent_names):
                    key = f"agent_{idx + 1}"
                    if env.agent_alive.get(name, False):
                        response.append((key, (("complete_rating", 0.0), "Timeout")))
                    else:
                        response.append(
                            (key, (("complete_rating", -1.0), "Eliminated"))
                        )
                return response
            return [("environment", (("terminated", True), "Max turns reached."))]

        if not env:
            return [("environment", (("terminated", False), ""))]

        # Check if final offers have been resolved
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


class DeadLastActionHandler(ActionHandler):
    """Handles actions for Dead Last."""

    def handle_action(
        self, env: SocialDeductionGame, agent_name: str, action: AgentAction
    ) -> None:
        if not isinstance(env, DeadLastEnv):
            return

        if action.action_type not in ["action", "speak"]:
            return

        if env.current_state == "Point":
            # "point NAME": the first roster name after 'point' (read_point), punctuation allowed ("point
            # Bob."). The pointer, an eliminated player or no name makes the point void, with a private
            # notice, instead of letting a later name count.
            alive_players = [n for n in env.agents if env.agent_alive.get(n, False)]
            target = read_point(action.argument, list(env.agents))

            def notice(text: str) -> None:
                env.recv_message("Environment", SimpleMessage(message=text), receivers=[agent_name])

            if target is None:
                notice(
                    f"[Game] {agent_name}: your point names no player after the word 'point', so it does not "
                    f"count. Alive players: [{', '.join(alive_players)}]."
                )
            elif target == agent_name:
                notice(f"[Game] {agent_name}: You cannot point at yourself, so your point does not count.")
            elif target not in alive_players:
                notice(
                    f"[Game] {agent_name}: your point names {target}, who is eliminated, so it does not count. "
                    f"Alive players: [{', '.join(alive_players)}]."
                )
            else:
                env.internal_state.setdefault("points", {})[agent_name] = target
                env.recv_message(
                    "Environment",
                    SimpleMessage(message=f"[Game] {agent_name} points at {target}."),
                )

        elif env.current_state == "Final_offer":
            # One whole number per survivor summing to 100, read by name when the offer names every
            # survivor and in seat order otherwise. An offer that cannot be read is kept as unreadable;
            # it counts as a mismatch when the offers are compared. The result line shows each reading.
            alive_players = [n for n in env.agents if env.agent_alive.get(n, False)]
            offer, problem, how = read_offer(action.argument, alive_players)
            env.internal_state.setdefault("offer_readings", {})[agent_name] = how
            if offer is not None:
                env.internal_state.setdefault("final_offers", {})[agent_name] = offer
            else:
                env.internal_state.setdefault("offer_problems", {})[agent_name] = (
                    f"'{action.argument.strip()[:80]}' {problem}"
                )

    def get_action_instruction(self, env: SocialDeductionGame, agent_name: str) -> str:
        if not isinstance(env, DeadLastEnv):
            return ""

        alive_players = [n for n in env.agents if env.agent_alive.get(n, False)]
        alive_str = ", ".join(alive_players)

        if env.current_state == "Point":
            return (
                f"You must point at someone to eliminate. Alive players: [{alive_str}]. "
                f"Use 'point NAME' (e.g., 'point Bob'). {point_rule(list(env.agents))} "
                f"You cannot point at yourself; a point that names you, an eliminated player or nobody does "
                f"not count, and you are told privately."
            )
        elif env.current_state == "Discussion":
            alive_count = len(alive_players)
            return (
                f"Discussion phase. {alive_count} players alive: [{alive_str}]. "
                f"Each living player speaks once, in seat order, and then everyone points. "
                f"Discuss who to target. A strict majority (>{alive_count // 2}) "
                f"pointing at the same person will eliminate them."
            )
        elif env.current_state == "Final_negotiation":
            return (
                f"FINAL NEGOTIATION! Only {len(alive_players)} players remain: [{alive_str}]. "
                f"Each survivor speaks once, in seat order; then everyone makes one binding final offer, at "
                f"the same time. "
                f"Negotiate how to split 100 points. You must agree on the exact split or everyone gets nothing. "
                f"Final offers list the shares in this order: [{alive_str}], unless every share is labelled "
                f"with its survivor's name."
            )
        elif env.current_state == "Final_offer":
            slots = ["XYZUVW"[i] if i < 6 else f"N{i + 1}" for i in range(len(alive_players))]
            meaning = ", ".join(f"{s} is {n}'s share" for s, n in zip(slots, alive_players))
            labelled = ", ".join(f"{n} {s}" for s, n in reversed(list(zip(slots, alive_players))))
            return (
                f"Submit your final offer. List {len(alive_players)} whole numbers that sum to 100, "
                f"one for each alive player in this order: [{alive_str}]. "
                f"Use 'offer {' '.join(slots)}' ({meaning}). "
                f"You may instead label every share with its survivor's name, in any order: "
                f"'offer {labelled}' means the same. {LABEL_RULE} "
                f"All players must submit identical splits or everyone gets nothing; an unclear offer, or one "
                f"that does not give exactly {len(alive_players)} whole numbers summing to 100, counts as a "
                f"mismatch, and the result line shows how every offer was read."
            )
        return ""


# ============================================================================
# Environment
# ============================================================================


class DeadLastEnv(SocialDeductionGame):
    """Dead Last game with voting, elimination, and final split negotiation."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(action_handler=DeadLastActionHandler(), **kwargs)

    @property
    def game_variant(self) -> str:
        """The rule variant this game plays: '' for the published game, else the name apply_variant gave it."""
        return str(self._config.get("variant", ""))

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
            "points": {},
            "round": 1,
            "game_over": False,
            "final_scores": {},
            "end_reason": "",
        }
        return obs

    def _check_eliminations(self) -> None:
        """Apply eliminations based on pointing votes."""
        if not self._should_transition_state():
            return

        if self.current_state == "Point":
            points = self.internal_state.get("points", {})
            if not points:
                # Every point named nobody living, the pointer, or was missing (each pointer whose point
                # did not count got a private notice): nobody is eliminated, and everyone is told so.
                self.recv_message(
                    "Environment",
                    SimpleMessage(
                        message="[Game] No point counted this round, so no one is eliminated."
                    ),
                )
                self.internal_state["round"] = self.internal_state.get("round", 1) + 1
            if points:
                # Count who each person pointed at
                vote_counts: Dict[str, int] = Counter(points.values())
                alive_count = sum(1 for a in self.agent_alive.values() if a)
                majority_threshold = alive_count // 2 + 1

                # Find target with strict majority
                eliminated = None
                for target, count in vote_counts.items():
                    if count >= majority_threshold:
                        eliminated = target
                        break

                if eliminated:
                    self.agent_alive[eliminated] = False
                    self.recv_message(
                        "Environment",
                        SimpleMessage(
                            message=f"[Game] {eliminated} was eliminated! "
                            f"({vote_counts[eliminated]} out of {alive_count} pointed at them)"
                        ),
                    )
                    _gen_logger.info(
                        f"{eliminated} was eliminated in round {self.internal_state.get('round', '?')}"
                    )
                else:
                    self.recv_message(
                        "Environment",
                        SimpleMessage(
                            message=f"[Game] No majority reached. No one is eliminated this round. "
                            f"(Votes: {dict(vote_counts)})"
                        ),
                    )

                # Clear points for next round
                self.internal_state["points"] = {}
                self.internal_state["round"] = self.internal_state.get("round", 1) + 1

                # Check if we should enter final phase
                alive_players = [
                    n for n in self.agents if self.agent_alive.get(n, False)
                ]
                if len(alive_players) <= 3:
                    # The no-discussion variant has no negotiation: the survivors make their offers right away,
                    # announced as the offers after a negotiation are.
                    negotiation = self.game_variant != NO_DISCUSSION
                    self.recv_message(
                        "Environment",
                        SimpleMessage(
                            message=f"[Game] Only {len(alive_players)} players remain: "
                            f"{', '.join(alive_players)}."
                            + (" Entering Final Negotiation!" if negotiation else "")
                        ),
                    )
                    # Override state to final negotiation
                    final_state = "Final_negotiation" if negotiation else "Final_offer"
                    self.current_state = final_state
                    if hasattr(self, "_state_turn_count"):
                        self._state_turn_count[final_state] = 0
                    if hasattr(self, "_round_robin_idx"):
                        self._round_robin_idx = 0
                    if not negotiation:
                        self.recv_message(
                            "Environment",
                            SimpleMessage(message="[Game] Entering state: Final_offer"),
                        )

        elif self.current_state == "Final_offer":
            # Offers are simultaneous and made once: every survivor acted in this step, so the
            # split is decided now. An unreadable offer (or none) cannot match, as in the told rule
            # "if they do not match, everyone gets nothing"; the result line names it.
            alive_players = [n for n in self.agents if self.agent_alive.get(n, False)]
            all_offers = self.internal_state.get("final_offers", {})
            offers = {n: all_offers[n] for n in alive_players if n in all_offers}
            problems = dict(self.internal_state.get("offer_problems", {}))
            for name in alive_players:
                if name not in offers and name not in problems:
                    problems[name] = "made no offer"

            # Every offer as the engine read it (by name or in seat order), so a misread is visible.
            readings = self.internal_state.get("offer_readings", {})
            parts = []
            for name in alive_players:
                if name in offers:
                    parts.append(
                        f"{name}'s offer ({readings.get(name, SEAT_ORDER)}): "
                        f"{format_split(offers[name], alive_players)}"
                    )
                elif problems[name] == "made no offer":
                    parts.append(f"{name} made no offer")
                else:
                    parts.append(f"{name}'s offer cannot be read as a split: {problems[name]}")
            as_read = "Final offers, as read: " + "; ".join(parts) + "."

            offer_values = list(offers.values())
            all_match = not problems and all(
                offer_values[0] == offer for offer in offer_values[1:]
            )

            if all_match and offer_values:
                agreed_split = offer_values[0]
                scores = {}
                for name in self.agents:
                    if self.agent_alive.get(name, False):
                        scores[name] = agreed_split.get(name, 0) / 100.0
                    else:
                        scores[name] = -1.0

                agreed = format_split(agreed_split, alive_players)
                self.internal_state["game_over"] = True
                self.internal_state["final_scores"] = scores
                self.internal_state["end_reason"] = (
                    f"{as_read} Players agreed on split: {agreed}!"
                )
                self.recv_message(
                    "Environment",
                    SimpleMessage(message=f"[Game] {as_read} Split agreed! {agreed}."),
                )
            else:
                # No agreement: eliminated players keep their -1.0; survivors get 0.0.
                scores = {}
                for name in self.agents:
                    scores[name] = 0.0 if self.agent_alive.get(name, True) else -1.0

                self.internal_state["game_over"] = True
                self.internal_state["final_scores"] = scores
                self.internal_state["end_reason"] = (
                    f"No agreement on split! {as_read} Everyone gets nothing."
                )
                self.recv_message(
                    "Environment",
                    SimpleMessage(
                        message=f"[Game] No agreement! {as_read} Everyone gets nothing."
                    ),
                )


# ============================================================================
# No-discussion variant (communication control, results/sec4_comm_control/README.md)
# ============================================================================

# SOTOPIA_GAME_VARIANT=no_discussion, read when the environment is created, removes the two speech states: each
# round is a point (Discussion goes), and the point that leaves three players opens the final offers at once
# (Final_negotiation goes; the line announcing the final no longer announces a negotiation). Points, the majority
# rule, offers and how they are read, the scoring, the turn cap and the other announcements stay as they are. The
# rules text loses the sentences about the speeches and says in one sentence that there is no discussion phase
# and no negotiation. Unset or empty, the game is the published one.
GAME_VARIANT_ENV = "SOTOPIA_GAME_VARIANT"
NO_DISCUSSION = "no_discussion"
# (published rules text, no-discussion rules text); each published text occurs once in config.json's description.
NO_DISCUSSION_RULES = [
    (
        "Each round, every living player speaks once, in seat order, and then all living players simultaneously "
        "point at someone to eliminate.",
        "Each round, all living players simultaneously point at someone to eliminate.",
    ),
    (
        "When three players remain, the game enters a Final Negotiation phase: each survivor speaks once, in seat "
        "order, and then the survivors simultaneously make one binding offer each, a split of 100 points.",
        "When three players remain, the survivors simultaneously make one binding offer each, a split of 100 points.",
    ),
    (
        "Eliminated players lose.\n\nDuring Discussion, use 'speak' to discuss strategy, form alliances, or argue "
        "for targets.\nDuring Point,",
        "Eliminated players lose. There is no discussion phase: each round goes straight to pointing, and the final "
        "offers are made without negotiation.\n\nDuring Point,",
    ),
    (
        "\nDuring Final_negotiation, use 'speak' to negotiate the split.\nDuring Final_offer,",
        "\nDuring Final_offer,",
    ),
]


def game_variant() -> str:
    """The variant SOTOPIA_GAME_VARIANT names: '' (the published game) or 'no_discussion'."""
    variant = os.environ.get(GAME_VARIANT_ENV, "").strip()
    if variant not in ("", NO_DISCUSSION):
        raise ValueError(
            f"{GAME_VARIANT_ENV}={variant!r}: Dead Last has no such variant (only {NO_DISCUSSION!r})"
        )
    return variant


def apply_variant(config: Dict[str, Any]) -> Dict[str, Any]:
    """`config` itself when no variant is set (or `config` is already the variant's). Otherwise a copy without
    Discussion and Final_negotiation (the game opens with Point, and Point leads to Point), with the no-discussion
    rules text, marked config["variant"]."""
    variant = game_variant()
    if not variant or config.get("variant") == variant:
        return config
    config = copy.deepcopy(config)
    text = config["description"]
    for published, replacement in NO_DISCUSSION_RULES:
        if text.count(published) != 1:
            raise ValueError(
                f"{variant}: the Dead Last rules text no longer has {published!r} exactly once"
            )
        text = text.replace(published, replacement)
    config["description"] = text
    for state in ("Discussion", "Final_negotiation"):
        _skip_state(config, state)
    config["variant"] = variant
    return config


def _skip_state(config: Dict[str, Any], state: str) -> None:
    """Take `state` out of the state machine: the state before it leads to the state after it."""
    transitions = config["state_transition"]
    after = transitions.pop(state)
    for before, nxt in transitions.items():
        if nxt == state:
            transitions[before] = after
    if config.get("initial_state") == state:
        config["initial_state"] = after
    config["state_properties"].pop(state)


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
) -> DeadLastEnv:
    """Create the Dead Last environment (the variant SOTOPIA_GAME_VARIANT names, if any)."""
    config = apply_variant(config)
    return DeadLastEnv(
        env_profile=env_profile,
        config=config,
        model_name=model_name,
        evaluators=[DeadLastEvaluator(max_turn_number=80)],
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
    config = apply_variant(config)
    agent_profiles = [ensure_agent_profile(entry) for entry in config.get("agents", [])]
    agent_goals = [
        config.get("role_goals", {}).get(entry.get("role", ""), "")
        for entry in config.get("agents", [])
    ]
    agent_names = [entry.get("name", "") for entry in config.get("agents", [])]
    scenario = config.get("description", "Dead Last").format(
        agent_names=", ".join(agent_names)
    )
    env_profile = EnvironmentProfile(
        scenario=scenario,
        relationship=RelationshipType.acquaintance,
        agent_goals=agent_goals,
        tag="dead_last",
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

    parser = argparse.ArgumentParser(description="Run Dead Last game.")
    parser.add_argument("--config", type=str, default=str(CONFIG_PATH))
    parser.add_argument("--roster", type=str, default=str(BASE_DIR / "roster.json"))
    args = parser.parse_args()

    config = load_config(args.config)
    roster = load_config(args.roster)
    config["agents"] = roster.get("agents", [])

    agent_model_name = get_model_names(config)
    env, agents = prepare_scenario("gpt-4o", agent_model_name, config)

    print("Dead Last")
    print("=" * 60)
    print_roster(config)
    print("=" * 60)

    variant = getattr(env, "game_variant", "")
    await arun_one_episode(
        env=env,
        agent_list=agents,
        omniscient=False,
        script_like=False,
        json_in_script=False,
        tag="test_dead_last",
        push_to_db=True,
        metadata={"variant": variant} if variant else None,
    )


if __name__ == "__main__":
    LOG_FILE = BASE_DIR / "dead_last_game_debug.log"
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
