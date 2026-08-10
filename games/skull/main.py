"""Launcher for the Skull (Skull & Roses) game."""

from __future__ import annotations

import asyncio
import os
import logging
import random
import re
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
        # Check turn limit first
        if turn_number >= self.max_turn_number:
            return [("environment", (("terminated", True), "Max turns reached"))]

        # Extract environment from kwargs
        env = kwargs.get("env")
        if not env:
            return [("environment", (("terminated", False), ""))]

        # Check game-specific win conditions
        terminated, reason, rewards = self._check_win_conditions(
            env, turn_number, messages
        )

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


# ============================================================================
# Action Handler
# ============================================================================


class SkullActionHandler(ActionHandler):
    def handle_action(
        self, env: SocialDeductionGame, agent_name: str, action: AgentAction
    ) -> None:
        if not isinstance(env, SkullEnv):
            return
        if action.action_type not in ["action", "speak"]:
            return

        arg = action.argument.strip().lower()

        if env.current_state == "Place":
            # If hand is empty, force a minimum bid to avoid infinite loop
            hand = env.internal_state["hands"].get(agent_name, [])
            if not hand and not re.search(r"bid\s*(\d+)", arg):
                arg = "bid 1"

            bid_match = re.search(r"bid\s*(\d+)", arg)
            if bid_match:
                # Player starts bidding -- only allowed if everyone placed at least 1
                all_placed = all(
                    len(env.internal_state["placed"].get(n, [])) >= 1
                    for n in env.agents
                    if env.agent_alive.get(n, False)
                )
                if all_placed:
                    bid_val = int(bid_match.group(1))
                    total_discs = sum(
                        len(env.internal_state["placed"].get(n, []))
                        for n in env.agents
                        if env.agent_alive.get(n, False)
                    )
                    bid_val = max(1, min(bid_val, total_discs))
                    env.internal_state["current_bid"] = bid_val
                    env.internal_state["current_bidder"] = agent_name
                    env.internal_state["bid_passed"] = set()
                    # Transition to Bid state
                    env.current_state = "Bid"
                    if hasattr(env, "_state_turn_count"):
                        env._state_turn_count["Bid"] = 0
                    if hasattr(env, "_round_robin_idx"):
                        env._round_robin_idx = 0
                    env.recv_message(
                        "Environment",
                        SimpleMessage(
                            message=f"[Game] {agent_name} starts bidding with bid {bid_val}! "
                            f"Entering Bid phase."
                        ),
                    )
            elif "skull" in arg:
                hand = env.internal_state["hands"].get(agent_name, [])
                if "skull" in hand:
                    hand.remove("skull")
                    env.internal_state["placed"].setdefault(agent_name, []).append(
                        "skull"
                    )
                    env.recv_message(
                        "Environment",
                        SimpleMessage(
                            message=f"[Place] {agent_name} placed a disc face-down. "
                            f"({len(env.internal_state['placed'][agent_name])} placed)"
                        ),
                    )
                else:
                    # No skull left, place rose
                    if "rose" in hand:
                        hand.remove("rose")
                        env.internal_state["placed"].setdefault(agent_name, []).append(
                            "rose"
                        )
                        env.recv_message(
                            "Environment",
                            SimpleMessage(
                                message=f"[Place] {agent_name} placed a disc face-down. "
                                f"({len(env.internal_state['placed'][agent_name])} placed)"
                            ),
                        )
            else:
                # Default: place rose
                hand = env.internal_state["hands"].get(agent_name, [])
                if "rose" in hand:
                    hand.remove("rose")
                    env.internal_state["placed"].setdefault(agent_name, []).append(
                        "rose"
                    )
                    env.recv_message(
                        "Environment",
                        SimpleMessage(
                            message=f"[Place] {agent_name} placed a disc face-down. "
                            f"({len(env.internal_state['placed'][agent_name])} placed)"
                        ),
                    )
                elif "skull" in hand:
                    hand.remove("skull")
                    env.internal_state["placed"].setdefault(agent_name, []).append(
                        "skull"
                    )
                    env.recv_message(
                        "Environment",
                        SimpleMessage(
                            message=f"[Place] {agent_name} placed a disc face-down. "
                            f"({len(env.internal_state['placed'][agent_name])} placed)"
                        ),
                    )

        elif env.current_state == "Bid":
            bid_match = re.search(r"bid\s*(\d+)", arg)
            if bid_match:
                bid_val = int(bid_match.group(1))
                current_bid = env.internal_state.get("current_bid", 0)
                total_discs = sum(
                    len(env.internal_state["placed"].get(n, []))
                    for n in env.agents
                    if env.agent_alive.get(n, False)
                )
                if bid_val > total_discs:
                    # Impossible bid — treat as pass
                    env.internal_state["bid_passed"].add(agent_name)
                    env.recv_message(
                        "Environment",
                        SimpleMessage(
                            message=f"[Bid] {agent_name} bid {bid_val}, which exceeds "
                            f"the {total_discs} discs on table. Counted as a pass."
                        ),
                    )
                elif bid_val > current_bid:
                    env.internal_state["current_bid"] = bid_val
                    env.internal_state["current_bidder"] = agent_name
                    env.recv_message(
                        "Environment",
                        SimpleMessage(message=f"[Bid] {agent_name} bids {bid_val}!"),
                    )
                else:
                    # Bid not higher — also treat as pass
                    env.internal_state["bid_passed"].add(agent_name)
                    env.recv_message(
                        "Environment",
                        SimpleMessage(
                            message=f"[Bid] {agent_name} bid {bid_val}, not higher than "
                            f"current bid {current_bid}. Counted as a pass."
                        ),
                    )
            elif "pass" in arg:
                env.internal_state["bid_passed"].add(agent_name)
                env.recv_message(
                    "Environment",
                    SimpleMessage(message=f"[Bid] {agent_name} passes."),
                )

        elif env.current_state == "Flip":
            bidder = env.internal_state.get("current_bidder")
            if agent_name != bidder:
                return
            # Parse "flip NAME"
            target = None
            for name in env.agents:
                if name.lower() in arg:
                    target = name
                    break
            # Skull rule: the bidder must flip ALL of their own placed discs
            # before flipping any other player's discs. If they pick someone
            # else while own discs remain, silently re-target to self so the
            # round advances by one flip instead of wasting an LLM turn.
            own_remaining = len(env.internal_state["placed"].get(bidder, []))
            if target and target != bidder and own_remaining > 0:
                target = bidder
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

            if not hand:
                return (
                    "You have no discs left to place. You MUST start bidding now: say 'bid N' "
                    "where N is how many discs you think you can flip safely."
                )

            bid_option = ""
            if all_placed_one:
                bid_option = (
                    " Or start bidding with 'bid N' (N = discs you can flip safely)."
                )

            return (
                f"Round wins: {round_wins} (need {wins_needed}). "
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
                f"Choose 'bid N' (must be > {current_bid}) or 'pass'."
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
            "game_over": False,
            "final_scores": {},
            "end_reason": "",
        }
        self._pending_init = False

        # Tell each player their hand privately
        for name in self.agents:
            self.recv_message(
                "Environment",
                SimpleMessage(
                    message="[Private] Your hand: 3 roses and 1 skull (4 discs total)."
                ),
                receivers=[name],
            )

        return obs

    def _update_action_mask(self) -> None:
        super()._update_action_mask()
        agents = list(self.agents)
        if self.current_state == "Bid":
            passed = self.internal_state.get("bid_passed", set())
            for idx, name in enumerate(agents):
                if name in passed:
                    self.action_mask[idx] = False
        elif self.current_state == "Flip":
            bidder = self.internal_state.get("current_bidder")
            for idx, name in enumerate(agents):
                self.action_mask[idx] = name == bidder

    def _check_eliminations(self) -> None:
        if not self._should_transition_state():
            return

        if self.current_state == "Bid":
            # Check if all but one have passed
            alive_players = [n for n in self.agents if self.agent_alive.get(n, False)]
            passed = self.internal_state.get("bid_passed", set())
            active_bidders = [n for n in alive_players if n not in passed]

            if len(active_bidders) <= 1 and self.internal_state.get("current_bidder"):
                bidder = self.internal_state["current_bidder"]
                bid = self.internal_state["current_bid"]
                # Transition to Flip
                self.current_state = "Flip"
                if hasattr(self, "_state_turn_count"):
                    self._state_turn_count["Flip"] = 0
                if hasattr(self, "_round_robin_idx"):
                    self._round_robin_idx = 0
                self.internal_state["flips_remaining"] = bid
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
                    removed = random.choice(bidder_hand)
                    bidder_hand.remove(removed)
                    self.recv_message(
                        "Environment",
                        SimpleMessage(
                            message=f"[Game] {bidder} loses a disc! "
                            f"({len(bidder_hand)} discs remaining)"
                        ),
                    )

                # Check if bidder eliminated
                if not bidder_hand:
                    self.agent_alive[bidder] = False
                    self.recv_message(
                        "Environment",
                        SimpleMessage(message=f"[Game] {bidder} is eliminated!"),
                    )

                self._start_new_round()

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
                    self._start_new_round()

    def _start_new_round(self) -> None:
        """Reset for a new round."""
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

        self.current_state = "Place"
        if hasattr(self, "_state_turn_count"):
            self._state_turn_count["Place"] = 0
        if hasattr(self, "_round_robin_idx"):
            self._round_robin_idx = 0

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
        evaluators=[SkullEvaluator(max_turn_number=120)],
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
