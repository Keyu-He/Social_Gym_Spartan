"""Launcher for the Skull (Skull & Roses) game."""

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
    graded_rank_rewards,
    load_config,
)
from sotopia.messages import AgentAction, Message, Observation, SimpleMessage
from sotopia.server import arun_one_episode

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"

# Turn limit. A round of four players takes about 15 to 25 turns, so 300 turns leave room for 12 or more rounds. It is
# a backstop against a game that never ends, so the models are not told about it.
MAX_TURNS = 300

os.environ.setdefault("REDIS_OM_URL", "redis://:@localhost:6379")
redis.Redis(host="localhost", port=6379)

_gen_logger = logging.getLogger("sotopia.generation")
_env_logger = logging.getLogger("sotopia.envs.social_game")


# ============================================================================
# Evaluator
# ============================================================================


class SkullEvaluator(SocialGameEndEvaluator):
    """Evaluator that checks Skull win conditions.

    Uses the _check_win_conditions pattern (same as WerewolfGameEndEvaluator)
    so the base-class __call__/__acall__ pipeline works correctly with the
    engine's evaluator aggregation.
    """

    def _check_win_conditions(  # type: ignore[override]
        self, env: Any, turn_number: int, messages: List[Tuple[str, Message]]
    ) -> Tuple[bool, str, Dict[str, float]]:
        """Check if game has ended based on Skull win conditions."""
        # 1. Check if _end_game() already flagged the game as over
        if env.internal_state.get("game_over"):
            final_scores: Dict[str, float] = env.internal_state.get("final_scores", {})
            reason: str = env.internal_state.get("end_reason", "Game over")
            return True, reason, final_scores

        # 2. Check round wins
        round_wins = env.internal_state.get("round_wins", {})
        wins_needed = env._config.get("wins_needed", 2)
        winner = None
        for name, wins in round_wins.items():
            if wins >= wins_needed:
                winner = name
                break

        # 3. Last-player-standing
        alive = [n for n in env.agents if env.agent_alive.get(n, False)]
        if not winner and len(alive) == 1:
            winner = alive[0]

        if winner:
            losers = [n for n in env.agents if n != winner]
            loser_score = -1.0 / len(losers) if losers else 0.0
            rewards: Dict[str, float] = {}
            for name in env.agents:
                rewards[name] = 1.0 if name == winner else loser_score
            win_reason = f"Game over! {winner} wins! Round wins: {round_wins}"
            return True, win_reason, rewards

        return False, "", {}

    def __call__(
        self, turn_number: int, messages: List[Tuple[str, Message]], **kwargs: Any
    ) -> List[Tuple[str, Tuple[Tuple[str, int | float | bool], str]]]:
        env = kwargs.get("env")
        if not env:
            if turn_number >= self.max_turn_number:
                return [("environment", (("terminated", True), "Max turns reached"))]
            return [("environment", (("terminated", False), ""))]

        # Check game-specific win conditions
        terminated, reason, rewards = self._check_win_conditions(
            env, turn_number, messages
        )
        if not terminated and turn_number >= self.max_turn_number:
            # Turn limit without a winner: rank by round wins, then by discs still owned (a disc is lost on
            # each failed challenge), with graded place rewards +1 / +1/3 / -1/3 / -1; ties share their places.
            owned = discs_owned(env)
            round_wins = env.internal_state.get("round_wins", {})
            score = {n: 100 * round_wins.get(n, 0) + owned[n] for n in env.agents}
            rewards = graded_rank_rewards(score)
            terminated = True
            reason = f"Max turns reached. Discs still owned: {owned}. Round wins: {round_wins}"
            env.recv_message("Environment", SimpleMessage(message=f"[Game] {reason}"))

        response: List[Tuple[str, Tuple[Tuple[str, int | float | bool], str]]] = [
            ("environment", (("terminated", terminated), reason))
        ]

        if terminated and rewards:
            agent_names = list(env.agents)
            for agent_name, reward in rewards.items():
                try:
                    idx = agent_names.index(agent_name)
                    generic_key = f"agent_{idx + 1}"
                    response.append((generic_key, (("complete_rating", reward), "")))
                except ValueError:
                    continue

        return response


def discs_owned(env: Any) -> Dict[str, int]:
    """Discs each player still owns: in hand, placed this round, or flipped this round (0 once eliminated)."""
    st = env.internal_state
    owned = {n: len(st.get("hands", {}).get(n, [])) + len(st.get("placed", {}).get(n, [])) for n in env.agents}
    for owner, _ in st.get("flipped_discs", []):
        owned[owner] = owned.get(owner, 0) + 1
    return {n: (owned[n] if env.agent_alive.get(n, False) else 0) for n in env.agents}


def _shared(name: str) -> Any:
    """A helper module shared by the games, loaded by path like the game modules themselves."""
    key = f"games_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, BASE_DIR.parent / f"{name}.py")
        assert spec and spec.loader
        sys.modules[key] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sys.modules[key])
    return sys.modules[key]


committed_move = _shared("committed_move")

# Move words. A disc named with a count ("3 roses and 1 skull") describes a hand, not a move; "Skull & Roses" is
# the game's name.
_ROSE = r"(?<!\d )(?<!\d)\brose\b"
_SKULL = r"(?<!\d )(?<!\d)\bskull\b(?!\s*(?:&|and)\s+roses)"
_BID = r"\bbid\s*:?\s*(\d+)\b"
_PASS = r"\bpass(?:es|ing)?\b"


def _same(m: Any) -> Any:
    return (m.move, m.match.groups())


def read_place(text: str) -> Tuple[str | None, int]:
    """The Place move a reply commits to: ('rose' | 'skull', 0), ('bid', N), ('unclear', 0) for two different
    moves, or (None, 0). Only committed moves count (committed_move): "I don't want to place my skull yet, so
    rose" is a rose, "I'll place a rose now and bid 2 next turn" is a rose."""
    move, mention, _ = committed_move.commitment(text, {"rose": _ROSE, "skull": _SKULL, "bid": _BID}, key=_same)
    if move == "bid" and mention is not None:
        return "bid", int(mention.match.group(1))
    return move, 0


def read_bid(text: str) -> Tuple[str | None, int]:
    """The Bid move a reply commits to: ('bid', N), ('pass', 0), ('unclear', 0) or (None, 0). "I could bid 5, but
    I'll pass" and "I'd rather pass than bid 6" are passes."""
    move, mention, _ = committed_move.commitment(text, {"bid": _BID, "pass": _PASS}, key=_same)
    if move == "bid" and mention is not None:
        return "bid", int(mention.match.group(1))
    return move, 0


_SELF = r"(?:mine|myself|my own(?:\s+(?:stack|discs?|pile))?|my (?:stack|discs?|pile)|own (?:stack|discs?))"


def read_flip(text: str, names: List[str], bidder: str | None, own_remaining: int) -> Tuple[str | None, bool]:
    """(the stack a Flip reply names or None, whether the reply names two different stacks to flip).

    A committed 'flip NAME' decides ("Alice flips Charlie's disc" is Charlie, "I'm not going to flip Bob; flip
    Diana" is Diana); 'flip mine' / 'my own' is the bidder's stack. Without one, the first player the reply names
    that is not negated ("Diana, not Bob" is Diana); once the bidder's own stack is empty, its own name counts
    only if no other player is named."""
    keys = {n.lower(): n for n in names}
    alts = "|".join(re.escape(k) for k in sorted(keys, key=len, reverse=True))
    verb = r"\bflip(?:s|ping)?\s+(?:the\s+)?(?:top\s+)?(?:disc\s+)?(?:of\s+|from\s+|on\s+)?"

    def target(m: Any) -> str | None:
        return keys.get(m.match.group(1), bidder)

    move, mention, _ = committed_move.commitment(text, {"flip": rf"{verb}({alts}|{_SELF})\b"}, key=target)
    if move == "unclear":
        return None, True
    if mention is not None:
        return target(mention), False
    named = [target(m) for m in committed_move.mentions(text, {"name": rf"\b({alts}|{_SELF})\b"}) if m.committed]
    named = [n for n in named if n]
    if own_remaining == 0 and any(n != bidder for n in named):
        named = [n for n in named if n != bidder]
    return (named[0] if named else None), False


def flip_target(text: str, names: List[str], bidder: str | None, own_remaining: int) -> str | None:
    """The stack a Flip reply names, or None (read_flip)."""
    return read_flip(text, names, bidder, own_remaining)[0]


# ============================================================================
# Action Handler
# ============================================================================


class SkullActionHandler(ActionHandler):
    def handle_action(
        self, env: SocialDeductionGame, agent_name: str, action: AgentAction
    ) -> None:
        if not isinstance(env, SkullEnv):
            return
        acting = agent_name == getattr(env, "_current_actor", None)
        if action.action_type not in ["action", "speak"]:
            if not acting:
                return
            # The acting player's reply failed or was 'none': handled below as a malformed move
            # (Place: the default disc; Bid: a pass; Flip: an invalid flip), each announced.
            arg = ""
        else:
            arg = action.argument.strip().lower()

        if env.current_state == "Place":
            self._handle_place(env, agent_name, arg)

        elif env.current_state == "Bid":
            move, bid_val = read_bid(arg)
            if move == "bid":
                current_bid = env.internal_state.get("current_bid", 0)
                total_discs = self._discs_on_table(env)
                if bid_val > total_discs:
                    # A bid above the discs on the table is lowered to that number, as the opening bid is
                    env.recv_message(
                        "Environment",
                        SimpleMessage(
                            message=f"[Bid] {agent_name} bid {bid_val}, more than the {total_discs} discs on the "
                            f"table, so the bid is {total_discs}."
                        ),
                    )
                    bid_val = total_discs
                if bid_val > current_bid:
                    env.internal_state["current_bid"] = bid_val
                    env.internal_state["current_bidder"] = agent_name
                    env.recv_message(
                        "Environment",
                        SimpleMessage(message=f"[Bid] {agent_name} bids {bid_val}!"),
                    )
                else:
                    # Bid not higher: treated as a pass
                    env.internal_state["bid_passed"].add(agent_name)
                    env.recv_message(
                        "Environment",
                        SimpleMessage(
                            message=f"[Bid] {agent_name} bid {bid_val}, not higher than "
                            f"current bid {current_bid}. Counted as a pass."
                        ),
                    )
            elif move == "pass":
                env.internal_state["bid_passed"].add(agent_name)
                env.recv_message(
                    "Environment",
                    SimpleMessage(message=f"[Bid] {agent_name} passes."),
                )
            else:
                # Neither a bid nor a pass, or both: counted as a pass, like a bid that is not higher
                env.internal_state["bid_passed"].add(agent_name)
                what = "gave an unclear reply (it commits to more than one move)" if move == "unclear" else (
                    "neither bid nor passed"
                )
                env.recv_message(
                    "Environment",
                    SimpleMessage(message=f"[Bid] {agent_name} {what}. Counted as a pass."),
                )

        elif env.current_state == "Flip":
            bidder = env.internal_state.get("current_bidder")
            if agent_name != bidder:
                return
            # Skull rule: the bidder must flip ALL of their own placed discs
            # before flipping any other player's discs. If they pick someone
            # else while own discs remain, re-target to self (the flip line names
            # the bidder's stack) so the round advances by one flip instead of
            # wasting an LLM turn.
            own_remaining = len(env.internal_state["placed"].get(bidder, []))
            target, unclear = read_flip(arg, list(env.agents), bidder, own_remaining)
            if target and target != bidder and own_remaining > 0:
                target = bidder
            if not (target and env.internal_state["placed"].get(target)):
                env.internal_state["invalid_flips"] = env.internal_state.get("invalid_flips", 0) + 1
                stacks = {n: len(v) for n, v in env.internal_state["placed"].items() if v}
                what = "named two different stacks to flip" if unclear else "named no stack with discs left"
                if env.internal_state["invalid_flips"] >= 2:
                    # Second invalid flip in this challenge: the challenge fails (cf. Liar's Dice, an illegal move costs a die)
                    env.recv_message(
                        "Environment",
                        SimpleMessage(
                            message=f"[Flip] {agent_name} again {what}. The challenge fails "
                            f"and {agent_name} loses a disc."
                        ),
                    )
                    env.internal_state["flip_result"] = "skull"
                    env.internal_state["skull_owner"] = None
                else:
                    env.recv_message(
                        "Environment",
                        SimpleMessage(
                            message=f"[Flip] {agent_name} {what}. Stacks: {stacks}. "
                            f"A second invalid flip fails the challenge."
                        ),
                    )
                return
            if target and env.internal_state["placed"].get(target):
                stack = env.internal_state["placed"][target]
                disc = stack.pop()  # Flip from top
                env.internal_state["flipped_discs"].append((target, disc))
                env.internal_state["flips_remaining"] -= 1
                env.internal_state["flipped_count"] += 1

                if disc == "skull":
                    # Hit a skull! Round over, bidder loses a disc
                    env.recv_message(
                        "Environment",
                        SimpleMessage(
                            message=f"[Flip] {agent_name} flipped {target}'s disc: SKULL! "
                            f"{agent_name} loses a disc."
                        ),
                    )
                    env.internal_state["flip_result"] = "skull"
                    env.internal_state["skull_owner"] = target
                else:
                    env.recv_message(
                        "Environment",
                        SimpleMessage(
                            message=f"[Flip] {agent_name} flipped {target}'s disc: ROSE. "
                            f"({env.internal_state['flips_remaining']} remaining)"
                        ),
                    )
                    if env.internal_state["flips_remaining"] <= 0:
                        env.internal_state["flip_result"] = "success"

    @staticmethod
    def _discs_on_table(env: "SkullEnv") -> int:
        return sum(len(env.internal_state["placed"].get(n, [])) for n in env.agents if env.agent_alive.get(n, False))

    @staticmethod
    def _put_down(env: "SkullEnv", agent_name: str, disc: str) -> None:
        env.internal_state["hands"][agent_name].remove(disc)
        env.internal_state["placed"].setdefault(agent_name, []).append(disc)
        env.recv_message(
            "Environment",
            SimpleMessage(
                message=f"[Place] {agent_name} placed a disc face-down. "
                f"({len(env.internal_state['placed'][agent_name])} placed)"
            ),
        )

    def _handle_place(self, env: "SkullEnv", agent_name: str, arg: str) -> None:
        """A Place reply: a disc, or a bid once every player has a disc down (a bid above the discs on the table
        opens at that number). A reply the engine cannot use (no disc and no bid, two different moves, a bid
        before every player has placed, a disc with an empty hand, or no reply) is announced and the player is
        asked once more; if the second reply cannot be used either, a disc drawn at random from the player's hand
        (from the roster seed) is placed, or with an empty hand the bidding opens at 1. Asking for a disc the
        player no longer holds places the other kind, and the player is told privately."""
        st = env.internal_state
        hand = st["hands"].get(agent_name, [])
        alive = [n for n in env.agents if env.agent_alive.get(n, False)]
        all_placed = all(len(st["placed"].get(n, [])) >= 1 for n in alive)
        move, asked = read_place(arg)

        problem = ""
        if not arg.strip():
            problem = "gave no reply"
        elif move == "unclear":
            problem = "is unclear (it commits to more than one move)"
        elif move is None:
            problem = "names no disc ('rose' or 'skull') and no bid"
        elif move == "bid" and not all_placed:
            problem = "bids before every player has placed a disc"
        elif move != "bid" and not hand:
            problem = "does not bid, but its hand is empty"
        if problem and not hand and not all_placed:
            # Cannot happen in clockwise play (everyone places before a player's hand runs out); kept as a guard
            st["place_retry"] = None
            env.recv_message(
                "Environment",
                SimpleMessage(
                    message=f"[Place] {agent_name} has no disc to place and cannot bid until every player "
                    f"has placed one. Turn passes."
                ),
            )
            return
        if problem:
            if st.get("place_retry") != agent_name:
                st["place_retry"] = agent_name
                env.recv_message(
                    "Environment",
                    SimpleMessage(message=f"[Place] {agent_name}'s reply {problem}. {agent_name} is asked once more."),
                )
                return
            st["place_retry"] = None
            if hand:
                disc = env.place_rng().choice(sorted(hand))
                env.recv_message(
                    "Environment",
                    SimpleMessage(
                        message=f"[Place] {agent_name}'s reply again {problem}, so a disc drawn at random from "
                        f"{agent_name}'s hand is placed."
                    ),
                )
                self._put_down(env, agent_name, disc)
                env.recv_message(
                    "Environment",
                    SimpleMessage(message=f"[Private] A {disc} was drawn at random from your hand and placed."),
                    receivers=[agent_name],
                )
                return
            env.recv_message(
                "Environment",
                SimpleMessage(
                    message=f"[Bid] {agent_name}'s reply again {problem}, and {agent_name} has no disc left to "
                    f"place, so the bid opens at 1."
                ),
            )
            move, asked = "bid", 1
        st["place_retry"] = None

        if move == "bid":
            total_discs = self._discs_on_table(env)
            bid_val = max(1, min(asked, total_discs))
            if bid_val != asked:
                env.recv_message(
                    "Environment",
                    SimpleMessage(
                        message=f"[Bid] {agent_name} asked to open at {asked}, but the bid must be between 1 and "
                        f"the {total_discs} discs on the table. It opens at {bid_val}."
                    ),
                )
            st["current_bid"] = bid_val
            st["current_bidder"] = agent_name
            st["bid_passed"] = set()
            # Transition to Bid state; the next bidder is the player after the opener (SkullEnv._advance)
            env.current_state = "Bid"
            env.recv_message(
                "Environment",
                SimpleMessage(message=f"[Game] {agent_name} starts bidding with bid {bid_val}! Entering Bid phase."),
            )
            return

        # Placing. Asking for a disc the player no longer holds places the other kind; the placer is told
        # privately (the disc's kind stays hidden from the others).
        assert move in ("rose", "skull")
        if move in hand:
            self._put_down(env, agent_name, move)
            return
        disc = "rose" if "rose" in hand else "skull"
        self._put_down(env, agent_name, disc)
        env.recv_message(
            "Environment",
            SimpleMessage(message=f"[Private] You have no {move} left in hand; a {disc} was placed instead."),
            receivers=[agent_name],
        )

    def get_action_instruction(self, env: SocialDeductionGame, agent_name: str) -> str:
        if not isinstance(env, SkullEnv):
            return ""
        # During initialization, internal_state may not be ready yet
        if getattr(env, "_pending_init", False) or not env.internal_state.get("hands"):
            return "Place a disc: 'rose' or 'skull'."

        round_wins = env.internal_state.get("round_wins", {})
        wins_needed = env._config.get("wins_needed", 2)
        alive_players = [n for n in env.agents if env.agent_alive.get(n, False)]

        if env.current_state == "Place":
            hand = env.internal_state.get("hands", {}).get(agent_name, [])
            placed = env.internal_state.get("placed", {})
            all_placed_one = all(len(placed.get(n, [])) >= 1 for n in alive_players)
            hand_desc = f"{hand.count('rose')}R/{hand.count('skull')}S"
            placed_counts = {n: len(placed.get(n, [])) for n in alive_players}

            retry = ""
            if env.internal_state.get("place_retry") == agent_name:
                fallback = "a disc drawn at random from your hand is placed" if hand else "the bidding opens at 1"
                retry = f"Your last reply could not be used; if this one cannot be used either, {fallback}. "
            if not hand:
                return (
                    f"{retry}You have no discs left to place. You MUST start bidding now: say 'bid N' "
                    "where N is how many discs you think you can flip safely."
                )

            bid_option = ""
            if all_placed_one:
                bid_option = (
                    " Or start bidding with 'bid N' (N = discs you can flip safely; at most the discs on the table)."
                )

            return (
                f"{retry}Round wins: {round_wins} (need {wins_needed}). "
                f"Your hand: [{hand_desc}]. Placed so far: {placed_counts}. "
                f"Place a disc: 'rose' or 'skull'.{bid_option}"
            )

        elif env.current_state == "Bid":
            current_bid = env.internal_state.get("current_bid", 0)
            bidder = env.internal_state.get("current_bidder", "")
            passed = env.internal_state.get("bid_passed", set())
            total_discs = sum(
                len(env.internal_state["placed"].get(n, [])) for n in alive_players
            )

            if agent_name in passed:
                return f"You already passed. Current bid: {current_bid} by {bidder}."

            return (
                f"Round wins: {round_wins}. Current bid: {current_bid} by {bidder}. "
                f"Total discs on table: {total_discs}. "
                f"Choose 'bid N' (must be > {current_bid}; a bid of {total_discs} ends the bidding at once, and a higher "
                f"one is lowered to {total_discs}) or 'pass' (a pass is final for this round; anything else counts as "
                f"a pass)."
            )

        elif env.current_state == "Flip":
            bidder = env.internal_state.get("current_bidder")
            if agent_name != bidder:
                return f"Waiting for {bidder} to flip discs."

            remaining = env.internal_state.get("flips_remaining", 0)
            placed = env.internal_state.get("placed", {})
            own_placed = len(placed.get(agent_name, []))
            available = {
                n: len(placed.get(n, [])) for n in alive_players if placed.get(n)
            }

            if own_placed > 0:
                return (
                    f"You must flip your own discs first! "
                    f"You have {own_placed} unflipped. {remaining} flips remaining. "
                    f"Say 'flip {agent_name}'."
                )
            else:
                return (
                    f"{remaining} flips remaining. Available stacks: {available}. "
                    f"Choose 'flip NAME' to flip the top disc of someone's stack."
                )
        return ""


# ============================================================================
# Environment
# ============================================================================


class SkullEnv(SocialDeductionGame):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(action_handler=SkullActionHandler(), **kwargs)

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, str] | None = None,
        agents: Agents | None = None,
        omniscient: bool = False,
        lite: bool = False,
        include_background_observations: bool = True,
    ) -> Dict[str, Observation]:
        # Pre-initialize internal_state so instructions are correct during super().reset()
        self._pending_init = True
        obs = super().reset(
            seed=seed,
            options=options,
            agents=agents,
            omniscient=omniscient,
            lite=lite,
            include_background_observations=include_background_observations,
        )
        self.internal_state = {
            "hands": {name: ["rose", "rose", "rose", "skull"] for name in self.agents},
            "placed": {name: [] for name in self.agents},
            "round_wins": {name: 0 for name in self.agents},
            "current_bid": 0,
            "current_bidder": None,
            "bid_passed": set(),
            "flips_remaining": 0,
            "flipped_count": 0,
            "flipped_discs": [],  # (owner, disc) tuples for returning after round
            "flip_result": None,
            "skull_owner": None,
            "next_actor": list(self.agents)[0],
            "place_retry": None,  # the player asked once more after a Place reply that could not be used
            "game_over": False,
            "final_scores": {},
            "end_reason": "",
        }
        self._pending_init = False
        # The lost disc is drawn from the roster seed, so a roster replays the same losses (an unseeded config
        # draws fresh)
        seed = self._config.get("seed")
        self._disc_rng = random.Random(f"{seed}-discs") if seed is not None else random.Random()
        # The disc placed for a player whose Place replies could not be used twice, from its own seeded stream
        self._place_rng = random.Random(f"{seed}-place") if seed is not None else random.Random()

        # Tell each player their hand privately
        for name in self.agents:
            self.recv_message(
                "Environment",
                SimpleMessage(
                    message="[Private] Your hand: 3 roses and 1 skull (4 discs total)."
                ),
                receivers=[name],
            )
        self._update_action_mask()  # the first player places first
        # The base reset built the first observations before the hands existed (a bare "Place a disc"): rebuild
        # the instructions and deliver the hand line and the game start now instead of one turn late.
        for name in self.agents:
            if name not in obs:
                continue
            if self.action_handler:
                obs[name].action_instruction = self.action_handler.get_action_instruction(self, name)
            obs[name].available_actions = self._get_available_actions(name)
            buffered = self.agent_message_buffer[name]
            if buffered:
                obs[name].last_turn = "\n".join(buffered)
                buffered.clear()

        return obs

    # Turn order (official rules): play goes clockwise in seat order, the bidding continues from the player after
    # the one who opened it, a pass is final for the round and passed players take no turns, the current high
    # bidder takes no turn while it is highest, the bidding ends when nobody else may raise or when a bid equals
    # the discs on the table, and the challenger places first in the next round (if the challenger was eliminated
    # by another player's skull, that player does). Malformed moves: a bidding move that is neither a raise nor a
    # pass counts as a pass; a flip naming no stack with discs gets a notice and a second one fails the challenge;
    # a bid before every player has placed places the default disc. The order is kept here in
    # internal_state["next_actor"] instead of the engine's round-robin counter, which restarts at the first seat
    # every few turns and gave turns to passed players.
    def _should_transition_state(self) -> bool:
        return False  # Skull moves between Place, Bid and Flip itself

    def place_rng(self) -> random.Random:
        if not hasattr(self, "_place_rng"):
            self._place_rng = random.Random()
        return self._place_rng

    def _alive(self) -> List[str]:
        return [n for n in self.agents if self.agent_alive.get(n, False)]

    def _seat_after(self, name: str, eligible: List[str]) -> str | None:
        order = list(self.agents)
        if not eligible:
            return None
        start = order.index(name) if name in order else -1
        for k in range(1, len(order) + 1):
            cand = order[(start + k) % len(order)]
            if cand in eligible:
                return cand
        return None

    def build_state(self, actions: Dict[str, AgentAction]) -> None:
        actor = next((n for n, m in zip(self.agents, self.action_mask) if m), None)
        self._current_actor = actor
        state_before = self.current_state
        super().build_state(actions)
        self._advance(actor, state_before)

    def _advance(self, actor: str | None, state_before: str) -> None:
        if self.internal_state.get("game_over") or actor is None:
            return
        if self.current_state == "Place" and state_before == "Place":
            if self.internal_state.get("place_retry") == actor:
                self.internal_state["next_actor"] = actor  # asked once more (SkullActionHandler._handle_place)
            else:
                self.internal_state["next_actor"] = self._seat_after(actor, self._alive())
        elif self.current_state == "Bid":
            self.internal_state["next_actor"] = self._seat_after(actor, self._may_bid())

    def _may_bid(self) -> List[str]:
        """Players who may still raise: alive, not passed, and not the current high bidder."""
        passed = self.internal_state.get("bid_passed", set())
        bidder = self.internal_state.get("current_bidder")
        return [n for n in self._alive() if n not in passed and n != bidder]

    def _update_action_mask(self) -> None:
        if not self.internal_state.get("hands"):  # during reset, before the game state exists
            super()._update_action_mask()
            return
        agents = list(self.agents)
        if self.current_state == "Flip":
            acting = self.internal_state.get("current_bidder")
        else:
            acting = self.internal_state.get("next_actor")
        self.action_mask = [name == acting for name in agents]

    def _check_eliminations(self) -> None:
        if self.current_state == "Bid":
            # The bidding ends when all but one have passed, or when the bid reaches the discs on the table
            alive_players = [n for n in self.agents if self.agent_alive.get(n, False)]
            total_discs = sum(len(self.internal_state["placed"].get(n, [])) for n in alive_players)
            at_max = self.internal_state.get("current_bid", 0) >= total_discs > 0

            if (not self._may_bid() or at_max) and self.internal_state.get("current_bidder"):
                bidder = self.internal_state["current_bidder"]
                bid = self.internal_state["current_bid"]
                # Transition to Flip
                self.current_state = "Flip"
                self.internal_state["flips_remaining"] = bid
                self.internal_state["invalid_flips"] = 0
                self.internal_state["flipped_count"] = 0
                self.internal_state["flip_result"] = None
                self.recv_message(
                    "Environment",
                    SimpleMessage(
                        message=f"[Game] {bidder} wins the bid at {bid}! "
                        f"Must now flip {bid} discs. Flip your own first!"
                    ),
                )

        elif self.current_state == "Flip":
            result = self.internal_state.get("flip_result")
            if result is None:
                return

            bidder = self.internal_state["current_bidder"]

            if result == "skull":
                # Bidder loses a random disc permanently
                # Return placed discs to hand first
                for name in self.agents:
                    placed = self.internal_state["placed"].get(name, [])
                    self.internal_state["hands"].setdefault(name, []).extend(placed)
                # Return flipped discs to their owners
                for owner, disc in self.internal_state.get("flipped_discs", []):
                    self.internal_state["hands"].setdefault(owner, []).append(disc)

                # Remove a random disc from bidder
                bidder_hand = self.internal_state["hands"].get(bidder, [])
                if bidder_hand:
                    removed = self._disc_rng.choice(bidder_hand)
                    bidder_hand.remove(removed)
                    self.recv_message(
                        "Environment",
                        SimpleMessage(
                            message=f"[Game] {bidder} loses a disc! "
                            f"({len(bidder_hand)} discs remaining)"
                        ),
                    )

                # Check if bidder eliminated
                starter = bidder
                if not bidder_hand:
                    self.agent_alive[bidder] = False
                    self.recv_message(
                        "Environment",
                        SimpleMessage(message=f"[Game] {bidder} is eliminated!"),
                    )
                    owner = self.internal_state.get("skull_owner")
                    alive = self._alive()
                    starter = owner if owner in alive and owner != bidder else self._seat_after(bidder, alive)

                self._start_new_round(starter)

            elif result == "success":
                # Bidder wins the round!
                self.internal_state["round_wins"][bidder] = (
                    self.internal_state["round_wins"].get(bidder, 0) + 1
                )
                wins = self.internal_state["round_wins"][bidder]
                wins_needed = self._config.get("wins_needed", 2)

                self.recv_message(
                    "Environment",
                    SimpleMessage(
                        message=f"[Game] {bidder} flipped all roses! "
                        f"Round win! ({wins}/{wins_needed})"
                    ),
                )

                if wins >= wins_needed:
                    self._end_game(bidder)
                else:
                    # Return placed and flipped discs, then start new round
                    for name in self.agents:
                        placed = self.internal_state["placed"].get(name, [])
                        self.internal_state["hands"].setdefault(name, []).extend(placed)
                    for owner, disc in self.internal_state.get("flipped_discs", []):
                        self.internal_state["hands"].setdefault(owner, []).append(disc)
                    self._start_new_round(bidder)

    def _start_new_round(self, starter: str | None = None) -> None:
        """Reset for a new round; `starter` places first."""
        alive = [n for n in self.agents if self.agent_alive.get(n, False)]

        # Check if only 1 player left
        if len(alive) <= 1:
            if alive:
                self._end_game(alive[0])
            return

        self.internal_state["placed"] = {name: [] for name in self.agents}
        self.internal_state["current_bid"] = 0
        self.internal_state["current_bidder"] = None
        self.internal_state["bid_passed"] = set()
        self.internal_state["flips_remaining"] = 0
        self.internal_state["flipped_count"] = 0
        self.internal_state["flipped_discs"] = []
        self.internal_state["flip_result"] = None
        self.internal_state["skull_owner"] = None
        self.internal_state["place_retry"] = None
        self.internal_state["next_actor"] = starter if starter in alive else alive[0]

        self.current_state = "Place"

        # Announce new round
        round_wins = self.internal_state.get("round_wins", {})
        hand_sizes = {n: len(self.internal_state["hands"].get(n, [])) for n in alive}
        self.recv_message(
            "Environment",
            SimpleMessage(
                message=f"[Game] New round! Wins: {round_wins}. "
                f"Discs remaining: {hand_sizes}."
            ),
        )

    def _end_game(self, winner: str) -> None:
        """End the game with a winner."""
        losers = [n for n in self.agents if n != winner]
        loser_score = -1.0 / len(losers) if losers else 0.0
        final = {}
        for name in self.agents:
            if name == winner:
                final[name] = 1.0
            else:
                final[name] = loser_score

        reason = (
            f"Game over! {winner} wins! "
            f"Round wins: {self.internal_state.get('round_wins', {})}"
        )
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
) -> SkullEnv:
    return SkullEnv(
        env_profile=env_profile,
        config=config,
        model_name=model_name,
        evaluators=[SkullEvaluator(max_turn_number=MAX_TURNS)],
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
    scenario = config.get("description", "Skull").format(
        agent_names=", ".join(agent_names)
    )
    env_profile = EnvironmentProfile(
        scenario=scenario,
        relationship=RelationshipType.acquaintance,
        agent_goals=agent_goals,
        tag="skull",
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

    parser = argparse.ArgumentParser(description="Run Skull game.")
    parser.add_argument("--config", type=str, default=str(CONFIG_PATH))
    parser.add_argument("--roster", type=str, default=str(BASE_DIR / "roster.json"))
    args = parser.parse_args()

    config = load_config(args.config)
    roster = load_config(args.roster)
    config["agents"] = roster.get("agents", [])

    agent_model_name = get_model_names(config)
    env, agents = prepare_scenario("gpt-4o", agent_model_name, config)

    print("Skull (Skull & Roses)")
    print("=" * 60)
    print_roster(config)
    print("=" * 60)

    await arun_one_episode(
        env=env,
        agent_list=agents,
        omniscient=False,
        script_like=False,
        json_in_script=False,
        tag="test_skull",
        push_to_db=True,
    )


if __name__ == "__main__":
    LOG_FILE = BASE_DIR / "skull_game_debug.log"
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
