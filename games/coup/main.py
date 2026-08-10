"""Launcher for the Coup social game scenario."""

from __future__ import annotations

import asyncio
import os
import logging
import random
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


# ============================================================================
# Evaluator
# ============================================================================


class CoupEvaluator(SocialGameEndEvaluator):
    """Evaluator that checks if only one player has influence remaining."""

    def __call__(
        self, turn_number: int, messages: List[Tuple[str, Message]], **kwargs: Any
    ) -> List[Tuple[str, Tuple[Tuple[str, int | float | bool], str]]]:
        if turn_number >= self.max_turn_number:
            env = kwargs.get("env")
            if env:
                response: List[
                    Tuple[str, Tuple[Tuple[str, int | float | bool], str]]
                ] = [("environment", (("terminated", True), "Max turns reached."))]
                hands = env.internal_state.get("hands", {})
                agent_names = list(env.agents)
                best = max((len(hands.get(n, [])) for n in agent_names), default=0)
                winners = [n for n in agent_names if len(hands.get(n, [])) == best]
                losers = [n for n in agent_names if n not in winners]
                win_score = 1.0 / len(winners) if winners and best > 0 else 0.0
                loser_score = -1.0 / len(losers) if losers else 0.0
                for idx, name in enumerate(agent_names):
                    key = f"agent_{idx + 1}"
                    if name in winners and best > 0:
                        response.append(
                            (
                                key,
                                (("complete_rating", win_score), "Winner (most cards)"),
                            )
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
        elif env.current_state == "Choose":
            self._handle_choose(env, agent_name, action)

    def _handle_declare(
        self, env: CoupEnv, agent_name: str, action: AgentAction
    ) -> None:
        arg = action.argument.strip().lower()
        coins = env.internal_state.get("coins", {})
        my_coins = coins.get(agent_name, 0)

        # Mandatory coup at 10+ coins
        if my_coins >= 10 and "coup" not in arg:
            env.recv_message(
                "Environment",
                SimpleMessage(
                    message=f"[Game] {agent_name}: You have {my_coins} coins. You MUST coup."
                ),
                receivers=[agent_name],
            )
            return

        # Parse the action
        parsed = self._parse_declaration(arg)
        if not parsed:
            env.recv_message(
                "Environment",
                SimpleMessage(
                    message=f"[Game] {agent_name}: Unrecognized action. "
                    f"Use: income, foreign_aid, tax, steal NAME, assassinate NAME, exchange, or coup NAME."
                ),
                receivers=[agent_name],
            )
            return

        action_type, target = parsed

        # Validate costs
        if action_type == "assassinate" and my_coins < 3:
            env.recv_message(
                "Environment",
                SimpleMessage(
                    message=f"[Game] {agent_name}: Assassinate costs 3 coins. You have {my_coins}."
                ),
                receivers=[agent_name],
            )
            return
        if action_type == "coup" and my_coins < 7:
            env.recv_message(
                "Environment",
                SimpleMessage(
                    message=f"[Game] {agent_name}: Coup costs 7 coins. You have {my_coins}."
                ),
                receivers=[agent_name],
            )
            return

        # Validate target exists and is alive
        if target:
            if target not in env.agents or not env.agent_alive.get(target, False):
                env.recv_message(
                    "Environment",
                    SimpleMessage(
                        message=f"[Game] {agent_name}: Invalid target '{target}'."
                    ),
                    receivers=[agent_name],
                )
                return
            if target == agent_name:
                env.recv_message(
                    "Environment",
                    SimpleMessage(
                        message=f"[Game] {agent_name}: You cannot target yourself."
                    ),
                    receivers=[agent_name],
                )
                return

        # Pay costs upfront (refunded if action is cancelled)
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

        if action_type == "coup":
            coins[agent_name] = my_coins - 7
            env.recv_message(
                "Environment",
                SimpleMessage(
                    message=f"[Game] {agent_name} launches a Coup against {target}! (paid 7 coins)"
                ),
            )
            self._lose_influence(env, target)
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

        if "challenge" in arg:
            env.internal_state["responses"][agent_name] = "challenge"
        elif "block" in arg:
            env.internal_state["responses"][agent_name] = "block"
        else:
            env.internal_state["responses"][agent_name] = "pass"

    def _handle_choose(
        self, env: CoupEnv, agent_name: str, action: AgentAction
    ) -> None:
        pending = env.internal_state.get("pending_exchange")
        if not pending or agent_name != pending.get("declarer"):
            return

        all_cards: List[str] = list(pending["all_cards"])
        keep_count: int = pending["keep_count"]
        valid_cards = sorted(set(all_cards))

        arg = action.argument.strip()
        parts = arg.split()
        # Accept "keep CARD..." or just "CARD..." (some models drop the verb)
        if parts and parts[0].lower() in ("keep", "exchange", "swap"):
            parts = parts[1:]

        # Match each token case-insensitively to a known card name
        chosen: List[str] = []
        lower_to_canonical = {c.lower(): c for c in valid_cards}
        for tok in parts:
            canonical = lower_to_canonical.get(tok.strip(",.").lower())
            if canonical is not None:
                chosen.append(canonical)
            if len(chosen) >= keep_count:
                break

        # Validate as a multiset subset of all_cards
        available = list(all_cards)
        kept: List[str] = []
        for c in chosen:
            if c in available:
                available.remove(c)
                kept.append(c)

        if len(kept) != keep_count:
            # Fall back to keeping the first keep_count cards (original hand prefix).
            kept = all_cards[:keep_count]
            available = all_cards[keep_count:]
            env.recv_message(
                "Environment",
                SimpleMessage(
                    message=(
                        f"[Game] {agent_name}: Could not parse choice from '{arg}'. "
                        f"Defaulted to keeping {kept}."
                    )
                ),
                receivers=[agent_name],
            )

        hands = env.internal_state.get("hands", {})
        deck = env.internal_state.get("deck", [])
        hands[agent_name] = kept
        deck.extend(available)
        random.shuffle(deck)

        env.recv_message(
            "Environment",
            SimpleMessage(
                message=f"[Private] You kept {kept}. Returned {available} to the deck."
            ),
            receivers=[agent_name],
        )
        env.recv_message(
            "Environment",
            SimpleMessage(message=f"[Game] {agent_name} completes the exchange."),
        )

        env.internal_state["pending_exchange"] = None

    def _parse_declaration(self, arg: str) -> Tuple[str, str | None] | None:
        """Parse a declaration string into (action_type, target)."""
        parts = arg.split()
        if not parts:
            return None

        action_word = parts[0]

        # Match action types
        action_map = {
            "income": "income",
            "foreign_aid": "foreign_aid",
            "foreignaid": "foreign_aid",
            "foreign": "foreign_aid",
            "tax": "tax",
            "steal": "steal",
            "assassinate": "assassinate",
            "exchange": "exchange",
            "coup": "coup",
        }

        action_type = action_map.get(action_word)
        if not action_type:
            # Try prefix matching
            for key, val in action_map.items():
                if key.startswith(action_word) and len(action_word) >= 3:
                    action_type = val
                    break

        if not action_type:
            return None

        # Extract target for targeted actions
        target = None
        if action_type in ["steal", "assassinate", "coup"]:
            # Find a capitalized name in remaining words
            for w in parts[1:]:
                if w[0].isupper() if w else False:
                    target = w
                    break
            # Also try lowercase matching
            if not target and len(parts) > 1:
                target = parts[1].capitalize()

        return (action_type, target)

    def _lose_influence(self, env: CoupEnv, player: str) -> None:
        """Force a player to lose one influence card."""
        hands = env.internal_state.get("hands", {})
        hand = hands.get(player, [])
        if hand:
            # Remove a card (the player would choose, but for simplicity, remove last)
            lost_card = hand.pop()
            revealed = env.internal_state.get("revealed", [])
            revealed.append(lost_card)
            env.recv_message(
                "Environment",
                SimpleMessage(
                    message=f"[Game] {player} reveals and loses: {lost_card}."
                ),
            )
            # Privately update remaining cards
            if hand:
                env.recv_message(
                    "Environment",
                    SimpleMessage(message=f"[Private] Your remaining card: {hand}"),
                    receivers=[player],
                )

        if len(hands.get(player, [])) == 0:
            env.agent_alive[player] = False
            env.recv_message(
                "Environment",
                SimpleMessage(
                    message=f"[Game] {player} has lost all influence and is eliminated!"
                ),
            )

    def _resolve_responses(self, env: CoupEnv) -> None:
        """Resolve all responses to a declaration."""
        declaration = env.internal_state.get("current_declaration")
        if not declaration:
            return

        responses = env.internal_state.get("responses", {})
        declarer = declaration["declarer"]
        action_type = declaration["action"]
        claimed_role = declaration.get("claimed_role")
        coins = env.internal_state.get("coins", {})
        hands = env.internal_state.get("hands", {})
        deck = env.internal_state.get("deck", [])

        # Priority: challenge > block > pass
        challengers = [n for n, r in responses.items() if r == "challenge"]
        blockers = [n for n, r in responses.items() if r == "block"]

        if challengers:
            challenger = challengers[0]  # First challenger
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
                self._lose_influence(env, challenger)

                # Declarer shuffles card back and draws new one
                declarer_hand.remove(claimed_role)
                deck.append(claimed_role)
                random.shuffle(deck)
                if deck:
                    new_card = deck.pop()
                    declarer_hand.append(new_card)
                    env.recv_message(
                        "Environment",
                        SimpleMessage(
                            message=f"[Private] You shuffled {claimed_role} back and drew: {new_card}. Your hand: {declarer_hand}"
                        ),
                        receivers=[declarer],
                    )

                # Action still takes effect
                self._apply_action(env, declaration)
            else:
                # Declarer doesn't have it - declarer loses
                env.recv_message(
                    "Environment",
                    SimpleMessage(
                        message=f"[Game] {challenger} challenges {declarer}'s claim of {claimed_role}. "
                        f"{declarer} does NOT have {claimed_role} - challenge SUCCEEDS! "
                        f"{declarer} loses influence."
                    ),
                )
                self._lose_influence(env, declarer)
                # Action cancelled

                # Refund costs
                if action_type == "assassinate":
                    coins[declarer] = coins.get(declarer, 0) + 3

        elif blockers:
            blocker = blockers[0]
            # Simplified: block succeeds without counter-challenge
            env.recv_message(
                "Environment",
                SimpleMessage(
                    message=f"[Game] {blocker} blocks {declarer}'s {action_type}. Action cancelled."
                ),
            )
            # Refund costs
            if action_type == "assassinate":
                coins[declarer] = coins.get(declarer, 0) + 3
        else:
            # All passed - action takes effect
            self._apply_action(env, declaration)

        # Clear declaration
        env.internal_state["current_declaration"] = None
        env.internal_state["responses"] = {}

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
            if target:
                # Coins already paid upfront in _handle_declare
                env.recv_message(
                    "Environment",
                    SimpleMessage(
                        message=f"[Game] {declarer} assassinates {target}! (paid 3 coins)"
                    ),
                )
                self._lose_influence(env, target)
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
                        f"Cards must be from: {sorted(set(all_cards))}."
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
                f"assassinate NAME (Assassin, 3 coins), exchange (Ambassador), coup NAME (7 coins)."
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
                can_block = action_type in ACTION_BLOCKERS

                return (
                    f"{declarer} declared {action_type}{target_str}{claim_str}. "
                    f"Your cards: {my_hand}. "
                    f"Choose: 'challenge' (risk a life to call bluff), "
                    + (
                        "'block' (claim you have a blocking role), "
                        if can_block
                        else ""
                    )
                    + "or 'pass'."
                )
        elif env.current_state == "Choose":
            pending = env.internal_state.get("pending_exchange") or {}
            if pending.get("declarer") == agent_name:
                all_cards = pending.get("all_cards", [])
                keep_count = pending.get("keep_count", 0)
                return (
                    f"Exchange: choose from {all_cards}. Keep exactly {keep_count}. "
                    f"Reply with 'keep CARD' (1 influence) or 'keep CARD1 CARD2' (2 influence)."
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

        # Build deck
        deck = card_types * cards_per_type
        random.shuffle(deck)

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

        return obs

    def _update_action_mask(self) -> None:
        """Override to handle declarer rotation and Response exclusion."""
        super()._update_action_mask()

        if self.current_state == "Declare":
            # Only the current declarer can act (like Resistance leader pattern)
            declarer_idx = self.internal_state.get("declarer_idx", 0)
            agent_list = list(self.agents)
            alive_indices = [
                i
                for i in range(len(agent_list))
                if self.agent_alive.get(agent_list[i], False)
            ]
            if alive_indices:
                actual_idx = alive_indices[declarer_idx % len(alive_indices)]
                self.action_mask = [False] * len(agent_list)
                self.action_mask[actual_idx] = True

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

        elif self.current_state == "Choose":
            # Only the declarer (with pending exchange) can act.
            pending = self.internal_state.get("pending_exchange") or {}
            declarer = pending.get("declarer", "")
            agent_list = list(self.agents)
            self.action_mask = [False] * len(agent_list)
            if declarer in agent_list:
                self.action_mask[agent_list.index(declarer)] = True

    def _perform_transition_state(self) -> None:
        """Route Response -> Choose when an exchange is pending; otherwise default."""
        if self.current_state == "Response" and self.internal_state.get(
            "pending_exchange"
        ):
            self.current_state = "Choose"
            if hasattr(self, "_state_turn_count"):
                self._state_turn_count[self.current_state] = 0
            if hasattr(self, "_round_robin_idx"):
                self._round_robin_idx = 0
            self.recv_message(
                "Environment",
                SimpleMessage(message=f"[Game] Entering state: {self.current_state}"),
            )
            return
        super()._perform_transition_state()

    def _check_eliminations(self) -> None:
        """Resolve responses when leaving Response state, then check for dead players."""
        # Resolve responses before transitioning out of Response
        if self.current_state == "Response" and self._should_transition_state():
            if self.internal_state.get("skip_response", False):
                # Income/coup: nothing to resolve, just clear the flag
                self.internal_state["skip_response"] = False
            else:
                handler = self.action_handler
                if isinstance(handler, CoupActionHandler):
                    handler._resolve_responses(self)
            # Advance to next declarer
            self.internal_state["declarer_idx"] = (
                self.internal_state.get("declarer_idx", 0) + 1
            )

        # Defensive cleanup: if leaving Choose with pending_exchange unresolved
        # (e.g., agent failed to respond), apply default keep-first-N and clear.
        if self.current_state == "Choose" and self._should_transition_state():
            pending = self.internal_state.get("pending_exchange")
            if pending:
                declarer = pending["declarer"]
                all_cards = list(pending["all_cards"])
                keep_count = pending["keep_count"]
                hands = self.internal_state.get("hands", {})
                deck = self.internal_state.get("deck", [])
                hands[declarer] = all_cards[:keep_count]
                deck.extend(all_cards[keep_count:])
                random.shuffle(deck)
                self.recv_message(
                    "Environment",
                    SimpleMessage(
                        message=f"[Game] {declarer} did not specify a choice; defaulted to keeping {hands[declarer]}."
                    ),
                )
                self.internal_state["pending_exchange"] = None

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
        evaluators=[CoupEvaluator(max_turn_number=100)],
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
