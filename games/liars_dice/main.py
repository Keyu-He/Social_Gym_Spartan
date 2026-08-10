"""Launcher for the Liar's Dice game."""

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
                agent_names = list(env.agents)
                dice = env.internal_state.get("dice", {})
                best_count = -1
                winner_name = None
                for name in agent_names:
                    count = len(dice.get(name, []))
                    if count > best_count:
                        best_count = count
                        winner_name = name
                losers = [n for n in agent_names if n != winner_name]
                loser_score = -1.0 / len(losers) if losers else 0.0
                for idx, name in enumerate(agent_names):
                    key = f"agent_{idx + 1}"
                    if name == winner_name and best_count > 0:
                        response.append(
                            (key, (("complete_rating", 1.0), "Winner (most dice)"))
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


class LiarsDiceActionHandler(ActionHandler):
    """Handles bidding and liar-calling actions."""

    def handle_action(
        self, env: SocialDeductionGame, agent_name: str, action: AgentAction
    ) -> None:
        if action.action_type not in ["action", "speak"]:
            return
        if not isinstance(env, LiarsDiceEnv):
            return

        arg = action.argument.strip().lower()

        # Handle "liar" call
        if "liar" in arg and "bid" not in arg:
            current_bid = env.internal_state.get("current_bid")
            if current_bid is None:
                env.recv_message(
                    "Environment",
                    SimpleMessage(
                        message=f"[Game] {agent_name}: No current bid to challenge. You must bid."
                    ),
                    receivers=[agent_name],
                )
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

            # Remove a die from the loser
            loser_dice = dice.get(loser, [])
            if loser_dice:
                loser_dice.pop()
                dice[loser] = loser_dice

            if len(dice.get(loser, [])) == 0:
                env.agent_alive[loser] = False
                env.recv_message(
                    "Environment",
                    SimpleMessage(
                        message=f"[Game] {loser} has lost all dice and is eliminated!"
                    ),
                )

            # Reset for new round
            env.internal_state["current_bid"] = None
            env.internal_state["round"] = env.internal_state.get("round", 0) + 1
            env.internal_state["needs_reroll"] = True

            # Determine next start player
            if env.agent_alive.get(loser, False):
                next_start = loser
            elif env.agent_alive.get(agent_name, False):
                next_start = agent_name
            else:
                next_start = None
                for pname in env.agents:
                    if env.agent_alive.get(pname, False):
                        next_start = pname
                        break
            env.internal_state["next_start_player"] = next_start

            # Reset turn counter so the self-loop state doesn't transition
            if hasattr(env, "_state_turn_count"):
                env._state_turn_count[env.current_state] = 0

            # Set round-robin index to the next start player
            if next_start:
                eligible_indices = [
                    idx
                    for idx, name in enumerate(env.agents)
                    if env.agent_alive.get(name, False)
                ]
                try:
                    target_idx = list(env.agents).index(next_start)
                    rr_pos = eligible_indices.index(target_idx)
                    env._round_robin_idx = rr_pos
                except (ValueError, IndexError):
                    env._round_robin_idx = 0
            else:
                env._round_robin_idx = 0
            return

        # Handle "bid QUANTITY FACE"
        if "bid" in arg:
            parts = arg.split()
            bid_idx = None
            for i, p in enumerate(parts):
                if p == "bid":
                    bid_idx = i
                    break

            if bid_idx is not None and bid_idx + 2 < len(parts):
                try:
                    quantity = int(parts[bid_idx + 1])
                    face = int(parts[bid_idx + 2])
                except (ValueError, IndexError):
                    env.recv_message(
                        "Environment",
                        SimpleMessage(
                            message=f"[Game] {agent_name}: Invalid bid. Use 'bid QUANTITY FACE' (e.g., 'bid 3 4')."
                        ),
                        receivers=[agent_name],
                    )
                    return

                if face < 1 or face > 6:
                    env.recv_message(
                        "Environment",
                        SimpleMessage(
                            message=f"[Game] {agent_name}: Face must be 1-6."
                        ),
                        receivers=[agent_name],
                    )
                    return

                if quantity < 1:
                    env.recv_message(
                        "Environment",
                        SimpleMessage(
                            message=f"[Game] {agent_name}: Quantity must be at least 1."
                        ),
                        receivers=[agent_name],
                    )
                    return

                current_bid = env.internal_state.get("current_bid")
                if current_bid is not None:
                    curr_q = current_bid["quantity"]
                    curr_f = current_bid["face"]
                    if quantity < curr_q or (quantity == curr_q and face <= curr_f):
                        env.recv_message(
                            "Environment",
                            SimpleMessage(
                                message=f"[Game] {agent_name}: Bid ({quantity}x {face}s) must be higher than "
                                f"current ({curr_q}x {curr_f}s). Increase quantity or same quantity with higher face."
                            ),
                            receivers=[agent_name],
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

        # Unrecognized
        env.recv_message(
            "Environment",
            SimpleMessage(
                message=f"[Game] {agent_name}: Unrecognized action '{action.argument}'. "
                f"Use 'bid QUANTITY FACE' or 'liar'."
            ),
            receivers=[agent_name],
        )

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
        self.internal_state = {
            "dice": {},
            "current_bid": None,
            "round": 1,
            "needs_reroll": False,
            "next_start_player": None,
        }
        for agent_name in self.agents:
            self.internal_state["dice"][agent_name] = [
                random.randint(1, 6) for _ in range(starting_dice)
            ]
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

    def _reroll_all_dice(self) -> None:
        dice = self.internal_state.get("dice", {})
        round_num = self.internal_state.get("round", 1)

        for agent_name in self.agents:
            if self.agent_alive.get(agent_name, False):
                num_dice = len(dice.get(agent_name, []))
                if num_dice > 0:
                    dice[agent_name] = [random.randint(1, 6) for _ in range(num_dice)]
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
        evaluators=[LiarsDiceEvaluator(max_turn_number=100)],
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
