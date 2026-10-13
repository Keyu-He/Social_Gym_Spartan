"""Launcher for the Public Goods Game."""

from __future__ import annotations

import asyncio
import os
import logging
import random
import re
from fractions import Fraction
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
    rank_split_rewards,
)
from sotopia.messages import AgentAction, Message, Observation, SimpleMessage
from sotopia.server import arun_one_episode

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"

os.environ.setdefault("REDIS_OM_URL", "redis://:@localhost:6379")
redis.Redis(host="localhost", port=6379)

_gen_logger = logging.getLogger("sotopia.generation")
_env_logger = logging.getLogger("sotopia.envs.social_game")


class PublicGoodsEvaluator(SocialGameEndEvaluator):
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


_NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
_ORDINALS = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th", 5: "5th", 6: "6th"}


def _tie_example(totals: List[int]) -> str:
    """What `rank_split_rewards` gives these totals, in words ("the player on 30 wins, ...")."""
    rewards = rank_split_rewards({str(i): t for i, t in enumerate(totals)})
    verbs = {1.0: ("wins", "win"), 0.0: ("draws", "draw"), -1.0: ("loses", "lose")}
    parts = []
    for total in sorted(set(totals), reverse=True):
        count = totals.count(total)
        verb = verbs[rewards[str(totals.index(total))]][0 if count == 1 else 1]
        who = "the player" if count == 1 else f"the {_NUMBER_WORDS[count]} players"
        parts.append(f"{who} on {total} {verb}")
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def tie_rule(n: int) -> str:
    """How `rank_split_rewards` treats equal totals at an n-player table, with two worked examples."""
    k = n // 2
    other = (
        f"one that holds {_ORDINALS[k + 1]} place or straddles the line between winners and losers"
        if n % 2
        else f"one that straddles the line between {_ORDINALS[k]} and {_ORDINALS[k + 1]} place"
    )
    a = [30] * (k + 1) + [12] * (n - k - 1)
    b = [30, 20, 20] + [12] * (n - 3)
    return (
        f"Players with equal totals share one result: a tied group that lies entirely within the top {k} places "
        f"wins, one that lies entirely within the bottom {k} places loses, and any other tied group ({other}) draws. "
        f"For example, with totals {', '.join(map(str, a))}, {_tie_example(a)}; with totals "
        f"{', '.join(map(str, b))}, {_tie_example(b)}."
    )


_AMOUNT = r"[-+]?\d+(?:\.\d+)?"


# An amount right after a negation in its clause ("I won't contribute 5") is not the contribution.
_NEGATION = re.compile(
    r"\b(?:not|no|never|neither|nor|cannot|don'?t|won'?t|can'?t|shouldn'?t|wouldn'?t)\b|n't\b"
)
_CLAUSE_BREAK = re.compile(r"[.;:!?,\n]|\bbut\b")


def is_negated(text: str, start: int) -> bool:
    """True when one of the 3 words before position `start`, in the same clause, is a negation."""
    clause = _CLAUSE_BREAK.split(text[:start])[-1]
    return bool(_NEGATION.search(" ".join(clause.split()[-3:])))


def read_contribution(argument: str, tokens: int = 10) -> int | None:
    """The tokens contributed, or None when the reply is unclear.

    The amount is the number after 'contribute' ("In round 3 I contribute 7" is 7), and a negated one
    ("I won't contribute 5") does not count; a reply without 'contribute' must hold exactly one number ("7").
    Two different amounts, no amount, a fraction of a token, or an amount outside 0..tokens is unclear,
    so the player is asked again instead of having the amount clamped.
    """
    text = argument.lower().replace("’", "'")
    stated = [
        (m.group(1), m.start()) for m in re.finditer(rf"\bcontribut\w*\s*[:=]?\s*({_AMOUNT})", text)
    ]
    if stated:
        values = {float(v) for v, at in stated if not is_negated(text, at)}
    else:
        values = {float(v) for v in re.findall(_AMOUNT, text)}
    if len(values) != 1:
        return None
    value = values.pop()
    if value != int(value) or not 0 <= value <= tokens:
        return None
    return int(value)


def amount(x: Fraction | int | float) -> str:
    """Display an exact amount, rounded to 3 decimals (exact for this game's eighths of a token)."""
    return f"{float(x):.3f}".rstrip("0").rstrip(".")


def amounts(values: Dict[str, Any]) -> str:
    return "{" + ", ".join(f"'{n}': {amount(v)}" for n, v in values.items()) + "}"


class PublicGoodsActionHandler(ActionHandler):
    def handle_action(
        self, env: SocialDeductionGame, agent_name: str, action: AgentAction
    ) -> None:
        if not isinstance(env, PublicGoodsEnv):
            return
        if env.current_state != "Contribute" or agent_name not in env.awaiting():
            return
        tokens = env._config.get("tokens_per_round", 10)
        read = (
            read_contribution(action.argument, tokens)
            if action.action_type in ["action", "speak"]
            else None
        )
        env.take_reply(agent_name, read)

    def get_action_instruction(self, env: SocialDeductionGame, agent_name: str) -> str:
        if not isinstance(env, PublicGoodsEnv):
            return ""
        if env.current_state == "Contribute":
            round_num = env.internal_state.get("round", 0) + 1
            max_rounds = env._config.get("max_rounds", 10)
            multiplier = env._config.get("multiplier", 1.5)
            scores = env.internal_state.get("scores", {})
            last_contribs = env.internal_state.get("last_contributions", {})
            contrib_info = ""
            if last_contribs:
                contrib_info = (
                    " Last round: "
                    + ", ".join(f"{n}: {c}" for n, c in last_contribs.items())
                    + "."
                )
            num_players = len(env.agents)
            top_n = num_players // 2
            text = (
                f"Round {round_num}/{max_rounds}. Earnings: {amounts(scores)}.{contrib_info} "
                f"Contribute 0-10 tokens. Use 'contribute N'. "
                f"Pool = total x {multiplier}, split equally. You keep uncontributed tokens. "
                f"Top {top_n} earners WIN, bottom {top_n} LOSE. Relative ranking matters. {tie_rule(num_players)}"
            )
            if env.internal_state.get("unclear", {}).get(agent_name):
                text += (
                    " Your last reply did not give one whole number of tokens from 0 to 10, so you are asked again."
                    " Reply 'contribute N' with N a whole number from 0 to 10."
                    " If this reply is also unclear, your contribution is drawn at random from 0 to 10."
                )
            return text
        return ""


class PublicGoodsEnv(SocialDeductionGame):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(action_handler=PublicGoodsActionHandler(), **kwargs)

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
            # Exact earnings (shares are eighths of a token); only the messages round them.
            "scores": {name: Fraction(0) for name in self.agents},
            "current_contributions": {},
            "unclear": {},
            "drawn": [],
            "last_contributions": {},
            "game_over": False,
            "final_scores": {},
            "end_reason": "",
        }
        # Contributions drawn for twice-unclear replies; seeded from the roster so replays agree.
        self._fallback_rng = random.Random(f"{self._config.get('seed')}-fallback")
        return obs

    def awaiting(self) -> List[str]:
        """Players whose contribution for the current round is not in yet (everyone, at the start of a round)."""
        contribs = self.internal_state.get("current_contributions", {})
        return [a for a in self.agents if a not in contribs]

    def take_reply(self, agent_name: str, contributed: int | None) -> None:
        """Hold a clear contribution; ask an unclear (or failed) reply once more, then draw it at random.

        A failure or an out-of-range amount never picks a fixed amount (0 would be the rank-dominant
        move). Contributions stay hidden until the round is complete: the state is private.
        """
        if contributed is not None:
            self.internal_state["current_contributions"][agent_name] = contributed
            return
        unclear = self.internal_state.setdefault("unclear", {})
        unclear[agent_name] = unclear.get(agent_name, 0) + 1
        round_num = self.internal_state.get("round", 0) + 1
        tokens = self._config.get("tokens_per_round", 10)
        if unclear[agent_name] == 1:
            message = (
                f"[Round {round_num}] {agent_name}'s reply did not give one whole number of tokens from 0 to {tokens}, "
                f"so {agent_name} is asked again. Contributions stay hidden until the round is complete."
            )
        else:
            self.internal_state["current_contributions"][agent_name] = (
                self._fallback_rng.randint(0, tokens)
            )
            self.internal_state.setdefault("drawn", []).append(agent_name)
            message = (
                f"[Round {round_num}] {agent_name}'s second reply was also unclear, "
                f"so {agent_name}'s contribution is drawn at random from 0 to {tokens}."
            )
        self.recv_message("Environment", SimpleMessage(message=message))

    def _update_action_mask(self) -> None:
        super()._update_action_mask()
        if (
            self.current_state == "Contribute"
            and "current_contributions" in self.internal_state
        ):
            due = self.awaiting()
            self.action_mask = [a in due for a in self.agents]

    def _check_eliminations(self) -> None:
        if self.current_state == "Contribute" and not self.awaiting():
            contribs = self.internal_state["current_contributions"]
            drawn = self.internal_state.get("drawn", [])
            scores = self.internal_state["scores"]

            tokens = self._config.get("tokens_per_round", 10)
            multiplier = self._config.get("multiplier", 1.5)
            num_players = len(self.agents)

            total_pool = sum(contribs.values()) * Fraction(str(multiplier))
            share = total_pool / num_players

            payoffs = {}
            for name in self.agents:
                contrib = contribs[name]
                payoff = (tokens - contrib) + share
                payoffs[name] = payoff
                scores[name] = scores.get(name, Fraction(0)) + payoff

            round_num = self.internal_state["round"] + 1
            self.internal_state["round"] = round_num
            self.internal_state["last_contributions"] = {
                n: contribs[n] for n in self.agents
            }
            self.internal_state["current_contributions"] = {}
            self.internal_state["unclear"] = {}
            self.internal_state["drawn"] = []

            def note(name: str) -> str:
                return " (drawn at random)" if name in drawn else ""

            contrib_str = ", ".join(
                f"{n}: contributed {contribs[n]}{note(n)}" for n in self.agents
            )
            payoff_str = ", ".join(
                f"{n}: earned {amount(payoffs[n])}" for n in self.agents
            )
            self.recv_message(
                "Environment",
                SimpleMessage(
                    message=f"[Round {round_num}] Contributions: {contrib_str}. "
                    f"Pool: {sum(contribs.values())} x {multiplier} = {amount(total_pool)}. "
                    f"Share: {amount(share)} each. Payoffs: {payoff_str}. "
                    f"Cumulative: {amounts(scores)}"
                ),
            )

            max_rounds = self._config.get("max_rounds", 10)
            if round_num >= max_rounds:
                # Top n/2 earners win, bottom n/2 lose; equal earners share a fate.
                final = rank_split_rewards(scores)

                reason = f"Game over after {max_rounds} rounds. Final earnings: {amounts(scores)}"
                self.internal_state["game_over"] = True
                self.internal_state["final_scores"] = final
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
) -> PublicGoodsEnv:
    return PublicGoodsEnv(
        env_profile=env_profile,
        config=config,
        model_name=model_name,
        # Safety cap only: every round takes one step, or two when a player is asked again.
        evaluators=[
            PublicGoodsEvaluator(max_turn_number=2 * config.get("max_rounds", 10))
        ],
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
    scenario = config.get("description", "Public Goods Game").format(
        agent_names=", ".join(agent_names)
    )
    env_profile = EnvironmentProfile(
        scenario=scenario,
        relationship=RelationshipType.acquaintance,
        agent_goals=agent_goals,
        tag="public_goods",
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

    parser = argparse.ArgumentParser(description="Run Public Goods Game.")
    parser.add_argument("--config", type=str, default=str(CONFIG_PATH))
    parser.add_argument("--roster", type=str, default=str(BASE_DIR / "roster.json"))
    args = parser.parse_args()
    config = load_config(args.config)
    roster = load_config(args.roster)
    config["agents"] = roster.get("agents", [])
    agent_model_name = get_model_names(config)
    env, agents = prepare_scenario("gpt-4o", agent_model_name, config)
    print("Public Goods Game")
    print("=" * 60)
    print_roster(config)
    print("=" * 60)
    await arun_one_episode(
        env=env,
        agent_list=agents,
        omniscient=False,
        script_like=False,
        json_in_script=False,
        tag="test_public_goods",
        push_to_db=True,
    )


if __name__ == "__main__":
    LOG_FILE = BASE_DIR / "public_goods_game_debug.log"
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
