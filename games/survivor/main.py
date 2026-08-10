"""Launcher for the Survivor game."""

from __future__ import annotations

import asyncio
import os
import logging
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
# Evaluator
# ============================================================================


class SurvivorEvaluator(SocialGameEndEvaluator):
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


class SurvivorActionHandler(ActionHandler):
    def handle_action(
        self, env: SocialDeductionGame, agent_name: str, action: AgentAction
    ) -> None:
        if not isinstance(env, SurvivorEnv):
            return
        if action.action_type not in ["action", "speak"]:
            return

        if env.current_state == "Vote":
            target = self._parse_vote(
                env, agent_name, action.argument, exclude_self=True
            )
            if target:
                env.internal_state.setdefault("votes", {})[agent_name] = target

        elif env.current_state == "Jury_vote":
            finalists = set(env.internal_state.get("finalists", []))
            target = self._parse_vote_from_set(action.argument, finalists)
            if target:
                env.internal_state.setdefault("jury_votes", {})[agent_name] = target

    def _parse_vote(
        self,
        env: SocialDeductionGame,
        voter: str,
        argument: str,
        exclude_self: bool = False,
    ) -> str | None:
        words = argument.split()
        for w in words:
            for name in env.agents:
                if w.lower() == name.lower():
                    if exclude_self and name == voter:
                        continue
                    if env.agent_alive.get(name, False):
                        return name
        return None

    def _parse_vote_from_set(self, argument: str, valid_names: set[str]) -> str | None:
        words = argument.split()
        for w in words:
            for name in valid_names:
                if w.lower() == name.lower():
                    return name
        return None

    def get_action_instruction(self, env: SocialDeductionGame, agent_name: str) -> str:
        if not isinstance(env, SurvivorEnv):
            return ""

        if env.current_state == "Discussion":
            alive = [n for n in env.agents if env.agent_alive.get(n, False)]
            round_num = env.internal_state.get("round", 0) + 1
            jury = env.internal_state.get("jury", [])
            return (
                f"Round {round_num}. {len(alive)} players remain: {', '.join(alive)}. "
                f"Jury so far: {', '.join(jury) if jury else 'none'}. "
                f"Discuss who to vote out. Remember: eliminated players become jurors "
                f"who pick the winner!"
            )

        elif env.current_state == "Vote":
            alive = [n for n in env.agents if env.agent_alive.get(n, False)]
            others = [n for n in alive if n != agent_name]
            return (
                f"Vote to eliminate someone. Use 'vote NAME'. "
                f"Options: {', '.join(others)}. You cannot vote for yourself."
            )

        elif env.current_state == "Jury_plea":
            finalists = env.internal_state.get("finalists", [])
            jury = env.internal_state.get("jury", [])
            if agent_name in finalists:
                return (
                    f"You are a finalist! Make your case to the jury "
                    f"({', '.join(jury)}) for why you should win. "
                    f"Other finalists: {', '.join(f for f in finalists if f != agent_name)}."
                )
            return "Waiting for finalists to make their pleas."

        elif env.current_state == "Jury_vote":
            finalists = env.internal_state.get("finalists", [])
            jury = env.internal_state.get("jury", [])
            if agent_name in jury:
                return (
                    f"As a juror, vote for who should win. "
                    f"Finalists: {', '.join(finalists)}. Use 'vote NAME'."
                )
            return "Waiting for jury to vote."

        return ""


# ============================================================================
# Environment
# ============================================================================


class SurvivorEnv(SocialDeductionGame):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(action_handler=SurvivorActionHandler(), **kwargs)

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
            "votes": {},
            "jury": [],
            "finalists": [],
            "jury_votes": {},
            "game_over": False,
            "final_scores": {},
            "end_reason": "",
        }
        return obs

    def _check_eliminations(self) -> None:
        if not self._should_transition_state():
            return

        if self.current_state == "Vote":
            votes = self.internal_state.get("votes", {})
            if not votes:
                return

            vote_counts = Counter(votes.values())
            max_votes = max(vote_counts.values())

            # Tie-break: first in agent order
            eliminated = None
            for name in self.agents:
                if vote_counts.get(name, 0) == max_votes:
                    eliminated = name
                    break

            if eliminated:
                self.agent_alive[eliminated] = False
                self.internal_state["jury"].append(eliminated)
                round_num = self.internal_state["round"] + 1
                self.internal_state["round"] = round_num
                self.internal_state["votes"] = {}

                vote_str = ", ".join(
                    f"{v}: {c} votes" for v, c in vote_counts.most_common()
                )
                self.recv_message(
                    "Environment",
                    SimpleMessage(
                        message=f"[Round {round_num}] Votes: {vote_str}. "
                        f"{eliminated} is eliminated and joins the jury!"
                    ),
                )

                alive = [n for n in self.agents if self.agent_alive.get(n, False)]
                if len(alive) <= 3:
                    self.internal_state["finalists"] = list(alive)

                    self.recv_message(
                        "Environment",
                        SimpleMessage(
                            message=f"[Game] Final {len(alive)} remain: {', '.join(alive)}! "
                            f"Jury: {', '.join(self.internal_state['jury'])}. "
                            f"Time for final pleas to the jury!"
                        ),
                    )

                    # Override state to Jury_plea
                    self.current_state = "Jury_plea"
                    if hasattr(self, "_state_turn_count"):
                        self._state_turn_count["Jury_plea"] = 0
                    if hasattr(self, "_round_robin_idx"):
                        self._round_robin_idx = 0

        elif self.current_state == "Jury_plea":
            # Jury_plea is about to transition to Jury_vote (via config).
            # Swap alive status: finalists -> dead, jurors -> alive.
            # This way the engine naturally allows only jurors to act in Jury_vote.
            finalists = self.internal_state.get("finalists", [])
            jury = self.internal_state.get("jury", [])

            for name in finalists:
                self.agent_alive[name] = False
            for name in jury:
                self.agent_alive[name] = True

            self.recv_message(
                "Environment",
                SimpleMessage(
                    message=f"[Game] Jury members ({', '.join(jury)}), cast your votes! "
                    f"Choose one finalist to win: {', '.join(finalists)}."
                ),
            )

        elif self.current_state == "Jury_vote":
            jury_votes = self.internal_state.get("jury_votes", {})
            finalists = self.internal_state.get("finalists", [])
            jury = self.internal_state.get("jury", [])

            if not jury_votes:
                return

            vote_counts = Counter(jury_votes.values())
            # The jury vote is the actual win condition. Reward shape:
            #   jury-vote winner(s): +1.0 (split if tied)
            #   other finalists (made the final but didn't win the vote): 0.0
            #   jurors/eliminated: -1.0
            top_count = vote_counts.most_common(1)[0][1] if vote_counts else 0
            winners = [n for n, c in vote_counts.items() if c == top_count and c > 0]
            if not winners:
                # No jury votes recorded -- fall back to all finalists tied.
                winners = list(finalists)
            win_score = 1.0 / len(winners) if winners else 0.0

            final_scores: Dict[str, float] = {}
            for name in self.agents:
                if name in winners:
                    final_scores[name] = win_score
                elif name in finalists:
                    final_scores[name] = 0.0
                else:
                    final_scores[name] = -1.0

            vote_str = ", ".join(
                f"{v}: {c} votes" for v, c in vote_counts.most_common()
            )
            winner_str = " & ".join(winners) if winners else "(none)"
            reason = f"Jury votes: {vote_str}. {winner_str} wins Survivor!"

            self.internal_state["game_over"] = True
            self.internal_state["final_scores"] = final_scores
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
) -> SurvivorEnv:
    return SurvivorEnv(
        env_profile=env_profile,
        config=config,
        model_name=model_name,
        evaluators=[SurvivorEvaluator(max_turn_number=80)],
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
    scenario = config.get("description", "Survivor").format(
        agent_names=", ".join(agent_names)
    )
    env_profile = EnvironmentProfile(
        scenario=scenario,
        relationship=RelationshipType.acquaintance,
        agent_goals=agent_goals,
        tag="survivor",
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

    parser = argparse.ArgumentParser(description="Run Survivor game.")
    parser.add_argument("--config", type=str, default=str(CONFIG_PATH))
    parser.add_argument("--roster", type=str, default=str(BASE_DIR / "roster.json"))
    args = parser.parse_args()
    config = load_config(args.config)
    roster = load_config(args.roster)
    config["agents"] = roster.get("agents", [])
    agent_model_name = get_model_names(config)
    env, agents = prepare_scenario("gpt-4o", agent_model_name, config)
    print("Survivor")
    print("=" * 60)
    print_roster(config)
    print("=" * 60)
    await arun_one_episode(
        env=env,
        agent_list=agents,
        omniscient=False,
        script_like=False,
        json_in_script=False,
        tag="test_survivor",
        push_to_db=True,
    )


if __name__ == "__main__":
    LOG_FILE = BASE_DIR / "survivor_game_debug.log"
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
