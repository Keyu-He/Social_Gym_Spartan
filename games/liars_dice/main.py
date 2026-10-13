"""Launcher for the Liar's Dice game."""

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

# Turn limit: a turn is one player's move. A game that reaches it goes to the most dice (see the evaluator). It is a
# backstop against a game that never ends, so the models are not told about it.
MAX_TURNS = 100

os.environ.setdefault("REDIS_OM_URL", "redis://:@localhost:6379")
redis.Redis(host="localhost", port=6379)

_gen_logger = logging.getLogger("sotopia.generation")
_env_logger = logging.getLogger("sotopia.envs.social_game")


# ============================================================================
# Evaluator
# ============================================================================


class LiarsDiceEvaluator(SocialGameEndEvaluator):
    """Evaluator that checks if only one player has dice remaining."""

    def __call__(
        self, turn_number: int, messages: List[Tuple[str, Message]], **kwargs: Any
    ) -> List[Tuple[str, Tuple[Tuple[str, int | float | bool], str]]]:
        if turn_number >= self.max_turn_number:
            env = kwargs.get("env")
            if env:
                response: List[
                    Tuple[str, Tuple[Tuple[str, int | float | bool], str]]
                ] = [("environment", (("terminated", True), "Max turns reached."))]
                # Most dice wins. Players tied for the most share the win as a
                # draw (equal rewards; the Bradley-Terry fit counts ties) rather
                # than letting seating order decide.
                agent_names = list(env.agents)
                dice = env.internal_state.get("dice", {})
                counts = {name: len(dice.get(name, [])) for name in agent_names}
                best_count = max(counts.values(), default=0)
                winners = [n for n in agent_names if counts[n] == best_count]
                losers = [n for n in agent_names if n not in winners]
                if not losers:
                    winner_score = 0.0
                    label = "Draw (all tied on dice)"
                elif len(winners) > 1:
                    winner_score = 1.0 / len(winners)
                    label = "Draw (tied for most dice)"
                else:
                    winner_score = 1.0
                    label = "Winner (most dice)"
                loser_score = -1.0 / len(losers) if losers else 0.0
                for idx, name in enumerate(agent_names):
                    key = f"agent_{idx + 1}"
                    if name in winners:
                        response.append(
                            (key, (("complete_rating", winner_score), label))
                        )
                    else:
                        response.append(
                            (key, (("complete_rating", loser_score), "Lost"))
                        )
                return response
            return [("environment", (("terminated", True), "Max turns reached."))]

        env = kwargs.get("env")
        if not env:
            return [("environment", (("terminated", False), ""))]

        alive_players = [n for n, alive in env.agent_alive.items() if alive]
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

# A call is the word 'liar' ("liar", "Liar!", "I call liar"), not "a liar", "liars" or "Liar's Dice". Perudo's
# "dudo", "challenge" and a bare "call" ("I call", "I call your bid") are the same call; "call for", "call out",
# "call off" and "call it a" are not.
_LIAR = (
    r"(?<!\ba )(?<!\ban )(?<!\bthe )\bliar\b(?!')|\bdudo\b|\bchallenge\b"
    r"|\bcall\b(?!\s+(?:for|out|off|it\s+an?)\b)"
)
# 'bid QUANTITY FACE'; "Bid: 3 4", "bid 3x4", "bid 3 4s" and "bid 3 of 4s" are the same bid, and number words are
# read as numbers ("bid three fours").
_BID = r"\bbid\s*:?\s*(-?\d+)\s*(?:x|\u00d7|\s+of\s+|\s)\s*(-?\d+)(?:'?s)?\b"
_NUMBER_WORDS = {
    w: str(i)
    for i, w in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen"
        " seventeen eighteen nineteen twenty".split()
    )
}
_FACE_WORDS = {"ones": "1s", "twos": "2s", "threes": "3s", "fours": "4s", "fives": "5s", "sixes": "6s"}


def _numbers_as_digits(text: str) -> str:
    """'bid three fours' -> 'bid 3 4s': quantity and face words become digits, so _BID reads them."""
    words = {**_NUMBER_WORDS, **_FACE_WORDS}
    return re.sub(
        r"\b(" + "|".join(sorted(words, key=len, reverse=True)) + r")\b",
        lambda m: words[m.group(1).lower()],
        text,
        flags=re.IGNORECASE,
    )


def read_move(text: str) -> Tuple[str, int, int]:
    """('liar', 0, 0), ('bid', quantity, face), ('unclear', 0, 0) for a reply that commits to two different
    moves, ('malformed', 0, 0) for a 'bid' without two numbers, or ('unrecognized', 0, 0).

    Only the move the reply commits to counts (committed_move): "Liar! I don't believe that bid.", "I call
    liar on your bid", "Dudo!", "I challenge that bid" and "I call" are liar calls, and "bid three fours" is
    bid 3 4; "I'm not going to call liar; bid 4 5", "Calling liar is too risky, so bid 4 5" and "bid 3 4, or
    should I call liar?" are bids; "I won't call liar" is no move.
    """
    text = _numbers_as_digits(text)
    move, mention, _ = committed_move.commitment(
        text, {"liar": _LIAR, "bid": _BID}, key=lambda m: (m.move, m.match.groups())
    )
    if move == "unclear":
        return "unclear", 0, 0
    if move == "liar":
        return "liar", 0, 0
    if move == "bid" and mention is not None:
        return "bid", int(mention.match.group(1)), int(mention.match.group(2))
    if committed_move.commitment(text, {"bid": r"\bbid\b"})[0] == "bid":
        return "malformed", 0, 0
    return "unrecognized", 0, 0



class LiarsDiceActionHandler(ActionHandler):
    """Handles bidding and liar-calling actions.

    On your turn you must raise the bid or call liar. Anything else (a bid that
    is not higher, a malformed command, calling liar with no bid on the table,
    or no move at all) is an illegal move and costs a die. Without that penalty
    a player who never makes a legal bid can never be challenged, never loses
    dice, and wins on the turn cap by default.

    Every lost die ends the round, as in the published game: after a challenge
    or an illegal move everyone re-rolls and the player who lost the die opens
    (the next living player in seat order if it was eliminated). So nobody is
    ever asked to raise or challenge their own bid, and no die leaves a hand
    while bids on that round's total are still standing.
    """

    def handle_action(
        self, env: SocialDeductionGame, agent_name: str, action: AgentAction
    ) -> None:
        if not isinstance(env, LiarsDiceEnv):
            return
        # Every agent submits an action each turn; only the one whose turn it
        # is can make a move, legal or otherwise. The rest send "none".
        if not self._is_acting(env, agent_name):
            return
        if action.action_type not in ["action", "speak"]:
            self._penalize(env, agent_name, "made no move")
            return

        move, quantity, face = read_move(action.argument)

        # Handle "liar" call
        if move == "liar":
            current_bid = env.internal_state.get("current_bid")
            if current_bid is None:
                self._penalize(env, agent_name, "called liar with no bid on the table")
                return

            bid_quantity = current_bid["quantity"]
            bid_face = current_bid["face"]
            bidder = current_bid["bidder"]

            # Count actual total of bid_face across all alive players' dice
            dice = env.internal_state.get("dice", {})
            actual_count = 0
            for pname, phand in dice.items():
                if env.agent_alive.get(pname, False):
                    actual_count += phand.count(bid_face)

            if actual_count < bid_quantity:
                loser = bidder
                result_msg = (
                    f"[Game] {agent_name} calls LIAR on {bidder}'s bid of "
                    f"{bid_quantity}x {bid_face}s! "
                    f"Actual count: {actual_count}. "
                    f"{agent_name} was RIGHT! {bidder} loses a die."
                )
            else:
                loser = agent_name
                result_msg = (
                    f"[Game] {agent_name} calls LIAR on {bidder}'s bid of "
                    f"{bid_quantity}x {bid_face}s! "
                    f"Actual count: {actual_count}. "
                    f"{agent_name} was WRONG! {agent_name} loses a die."
                )

            env.recv_message("Environment", SimpleMessage(message=result_msg))
            _gen_logger.info(result_msg)

            # Reveal all dice
            dice_reveal = "[Game] All dice revealed: "
            dice_parts = []
            for pname in env.agents:
                if env.agent_alive.get(pname, False):
                    dice_parts.append(f"{pname}: {dice.get(pname, [])}")
            dice_reveal += ", ".join(dice_parts)
            env.recv_message("Environment", SimpleMessage(message=dice_reveal))

            self._end_round(env, loser)
            return

        if move == "unclear":
            self._penalize(
                env,
                agent_name,
                f"gave an unclear move '{action.argument}' (it commits to more than one move)",
            )
            return

        # Handle "bid QUANTITY FACE"
        if move == "malformed":
            self._penalize(
                env,
                agent_name,
                f"made a malformed bid '{action.argument}' (use 'bid QUANTITY FACE')",
            )
            return
        if move == "bid":
            if face < 1 or face > 6:
                self._penalize(
                    env, agent_name, f"bid on face {face} (faces are 1-6)"
                )
                return

            if quantity < 1:
                self._penalize(env, agent_name, f"bid a quantity of {quantity}")
                return

            current_bid = env.internal_state.get("current_bid")
            if current_bid is not None:
                curr_q = current_bid["quantity"]
                curr_f = current_bid["face"]
                if quantity < curr_q or (quantity == curr_q and face <= curr_f):
                    self._penalize(
                        env,
                        agent_name,
                        f"bid {quantity}x {face}s, which is not higher than the current "
                        f"{curr_q}x {curr_f}s",
                    )
                    return

            env.internal_state["current_bid"] = {
                "quantity": quantity,
                "face": face,
                "bidder": agent_name,
            }
            env.recv_message(
                "Environment",
                SimpleMessage(
                    message=f"[Game] {agent_name} bids {quantity}x {face}s."
                ),
            )
            return

        self._penalize(
            env,
            agent_name,
            f"gave an unrecognized action '{action.argument}' (use 'bid QUANTITY FACE' or 'liar')",
        )

    @staticmethod
    def _is_acting(env: LiarsDiceEnv, agent_name: str) -> bool:
        """Whether it is this agent's turn (the mask is still this turn's)."""
        try:
            return bool(env.action_mask[list(env.agents).index(agent_name)])
        except (ValueError, IndexError):
            return False

    def _penalize(self, env: LiarsDiceEnv, agent_name: str, reason: str) -> None:
        """Charge a die for an illegal move; like a lost challenge, it ends the round."""
        env.recv_message(
            "Environment",
            SimpleMessage(
                message=f"[Game] {agent_name} {reason}. Illegal move: {agent_name} forfeits a die "
                f"and the round ends."
            ),
        )
        self._end_round(env, agent_name)

    def _end_round(self, env: LiarsDiceEnv, loser: str) -> None:
        """``loser`` loses a die; everyone re-rolls and the loser opens the next round.

        If the loser is eliminated, the next living player after it in seat order opens.
        The re-roll happens inside this step, so the observations built at the end of the
        step (including the next opener's "Your dice" instruction) already show the new
        round's dice; deferring the roll to the next step made openers bid on the previous
        round's dice.
        """
        self._remove_die(env, loser)
        env.internal_state["current_bid"] = None
        env.internal_state["round"] = env.internal_state.get("round", 0) + 1
        env._reroll_all_dice()
        env.internal_state["needs_reroll"] = False
        if env.agent_alive.get(loser, False):
            next_start = loser
        else:
            next_start = self._next_alive_after(env, loser)
        env.internal_state["next_start_player"] = next_start
        self._set_next_actor(env, next_start)

    @staticmethod
    def _remove_die(env: LiarsDiceEnv, agent_name: str) -> bool:
        """Take one die away; return True if that eliminated the player."""
        dice = env.internal_state.get("dice", {})
        hand = dice.get(agent_name, [])
        if hand:
            hand.pop()
            dice[agent_name] = hand
        if hand:
            return False
        env.agent_alive[agent_name] = False
        env.recv_message(
            "Environment",
            SimpleMessage(
                message=f"[Game] {agent_name} has lost all dice and is eliminated!"
            ),
        )
        return True

    @staticmethod
    def _next_alive_after(env: LiarsDiceEnv, agent_name: str) -> str | None:
        names = list(env.agents)
        start = names.index(agent_name)
        for offset in range(1, len(names) + 1):
            candidate = names[(start + offset) % len(names)]
            if env.agent_alive.get(candidate, False):
                return candidate
        return None

    @staticmethod
    def _set_next_actor(env: LiarsDiceEnv, agent_name: str | None) -> None:
        """Make ``agent_name`` act next without breaking the seating order.

        Bidding loops on itself, and the base class resets the round-robin
        pointer whenever the state's turn count reaches the number of living
        players. Keeping pointer and count equal makes that reset coincide with
        the pointer wrapping, so nobody is skipped or acts twice in a row.
        """
        alive = [name for name in env.agents if env.agent_alive.get(name, False)]
        pos = alive.index(agent_name) if agent_name in alive else 0
        env._round_robin_idx = pos
        if hasattr(env, "_state_turn_count"):
            env._state_turn_count[env.current_state] = pos

    def get_action_instruction(self, env: SocialDeductionGame, agent_name: str) -> str:
        if not isinstance(env, LiarsDiceEnv):
            return ""

        dice = env.internal_state.get("dice", {})
        my_dice = dice.get(agent_name, [])
        current_bid = env.internal_state.get("current_bid")

        total_dice = sum(
            len(d) for pname, d in dice.items() if env.agent_alive.get(pname, False)
        )

        player_info = []
        for pname in env.agents:
            if env.agent_alive.get(pname, False):
                player_info.append(f"{pname}: {len(dice.get(pname, []))} dice")

        instruction = (
            f"Your dice: {my_dice}. Total dice in play: {total_dice}. "
            f"Players: [{', '.join(player_info)}]. "
        )

        if current_bid is None:
            instruction += (
                "No current bid. You must place the first bid. "
                "Use 'bid QUANTITY FACE' (e.g., 'bid 2 3' means 'at least two 3s among all dice')."
            )
        else:
            instruction += (
                f"Current bid: {current_bid['quantity']}x {current_bid['face']}s "
                f"(by {current_bid['bidder']}). "
                f"Raise with 'bid QUANTITY FACE' or call 'liar' to challenge."
            )
        instruction += (
            " An illegal move (a bid that is not higher, a malformed command, a reply that commits to two"
            " different moves, or no move) costs you a die and ends the round."
        )

        return instruction


# ============================================================================
# Environment
# ============================================================================


class LiarsDiceEnv(SocialDeductionGame):
    """Liar's Dice with bidding, bluffing, and elimination."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(action_handler=LiarsDiceActionHandler(), **kwargs)
        self.internal_state: Dict[str, Any] = {
            "dice": {},
            "current_bid": None,
            "round": 0,
            "needs_reroll": True,
            "next_start_player": None,
        }

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
        starting_dice = self._config.get("starting_dice", 5)
        # Dice come from the roster seed, so a roster replays the same rolls (an unseeded config rolls fresh dice)
        seed = self._config.get("seed")
        self._dice_rng = random.Random(f"{seed}-dice") if seed is not None else random.Random()
        self.internal_state = {
            "dice": {},
            "current_bid": None,
            "round": 1,
            "needs_reroll": False,
            "next_start_player": None,
        }
        for agent_name in self.agents:
            self.internal_state["dice"][agent_name] = self._roll(starting_dice)
        # Privately reveal dice
        for agent_name in self.agents:
            my_dice = self.internal_state["dice"][agent_name]
            self.recv_message(
                "Environment",
                SimpleMessage(message=f"[Private] Round 1 - Your dice: {my_dice}"),
                receivers=[agent_name],
            )
        total_dice = sum(len(d) for d in self.internal_state["dice"].values())
        self.recv_message(
            "Environment",
            SimpleMessage(
                message=f"[Game] Round 1 begins. Total dice: {total_dice}. Bidding starts!"
            ),
        )
        # The base reset built each agent's action instruction before the dice existed
        # ("Your dice: []"); rebuild it so the round-1 opener bids on its real dice, and deliver
        # the private roll and the round start now instead of one turn late.
        for agent_name in self.agents:
            if agent_name not in obs:
                continue
            if self.action_handler:
                obs[agent_name].action_instruction = self.action_handler.get_action_instruction(
                    self, agent_name
                )
            buffered = self.agent_message_buffer[agent_name]
            if buffered:
                obs[agent_name].last_turn = "\n".join(buffered)
                buffered.clear()
        return obs

    async def astep(
        self, actions: Dict[str, AgentAction] | Dict[str, Dict[str, int | str]]
    ) -> Tuple[
        Dict[str, Any],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Any],
    ]:
        # Re-roll dice if needed (after liar resolution from previous turn)
        if self.internal_state.get("needs_reroll", False):
            self._reroll_all_dice()
            self.internal_state["needs_reroll"] = False

        return await super().astep(actions)

    def _roll(self, n: int) -> List[int]:
        if not hasattr(self, "_dice_rng"):
            self._dice_rng = random.Random()
        return [self._dice_rng.randint(1, 6) for _ in range(n)]

    def _reroll_all_dice(self) -> None:
        dice = self.internal_state.get("dice", {})
        round_num = self.internal_state.get("round", 1)

        for agent_name in self.agents:
            if self.agent_alive.get(agent_name, False):
                num_dice = len(dice.get(agent_name, []))
                if num_dice > 0:
                    dice[agent_name] = self._roll(num_dice)
                    self.recv_message(
                        "Environment",
                        SimpleMessage(
                            message=f"[Private] Round {round_num} - Your dice: {dice[agent_name]}"
                        ),
                        receivers=[agent_name],
                    )

        total_dice = sum(
            len(d) for pname, d in dice.items() if self.agent_alive.get(pname, False)
        )
        self.recv_message(
            "Environment",
            SimpleMessage(
                message=f"[Game] Round {round_num} begins. Total dice: {total_dice}. Bidding starts!"
            ),
        )

    def _check_eliminations(self) -> None:
        dice = self.internal_state.get("dice", {})
        for agent_name in self.agents:
            if self.agent_alive.get(agent_name, False):
                if len(dice.get(agent_name, [])) == 0:
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
) -> LiarsDiceEnv:
    return LiarsDiceEnv(
        env_profile=env_profile,
        config=config,
        model_name=model_name,
        evaluators=[LiarsDiceEvaluator(max_turn_number=int(config.get("max_turns", MAX_TURNS)))],
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
    scenario = config.get("description", "Liar's Dice").format(
        agent_names=", ".join(agent_names)
    )
    env_profile = EnvironmentProfile(
        scenario=scenario,
        relationship=RelationshipType.acquaintance,
        agent_goals=agent_goals,
        tag="liars_dice",
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

    parser = argparse.ArgumentParser(description="Run Liar's Dice game.")
    parser.add_argument("--config", type=str, default=str(CONFIG_PATH))
    parser.add_argument("--roster", type=str, default=str(BASE_DIR / "roster.json"))
    args = parser.parse_args()

    config = load_config(args.config)
    roster = load_config(args.roster)
    config["agents"] = roster.get("agents", [])

    agent_model_name = get_model_names(config)
    env, agents = prepare_scenario("gpt-4o", agent_model_name, config)

    print("Liar's Dice")
    print("=" * 60)
    print_roster(config)
    print("=" * 60)

    await arun_one_episode(
        env=env,
        agent_list=agents,
        omniscient=False,
        script_like=False,
        json_in_script=False,
        tag="test_liars_dice",
        push_to_db=True,
    )


if __name__ == "__main__":
    LOG_FILE = BASE_DIR / "liars_dice_game_debug.log"
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
