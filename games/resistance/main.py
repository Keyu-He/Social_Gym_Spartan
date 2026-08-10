"""Launcher for The Resistance social game scenario."""

from __future__ import annotations

import asyncio
import os
import logging
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


class ResistanceEvaluator(SocialGameEndEvaluator):
    """Evaluator for The Resistance win conditions."""

    def __call__(
        self, turn_number: int, messages: List[Tuple[str, Message]], **kwargs: Any
    ) -> List[Tuple[str, Tuple[Tuple[str, int | float | bool], str]]]:
        if turn_number >= self.max_turn_number:
            env = kwargs.get("env")
            if env:
                response: List[
                    Tuple[str, Tuple[Tuple[str, int | float | bool], str]]
                ] = [("environment", (("terminated", True), "Max turns reached."))]
                for idx, name in enumerate(env.agents):
                    response.append(
                        (f"agent_{idx + 1}", (("complete_rating", 0.0), "Timeout"))
                    )
                return response
            return [("environment", (("terminated", True), "Max turns reached."))]

        env = kwargs.get("env")
        if not env:
            return [("environment", (("terminated", False), ""))]

        if env.internal_state.get("game_over", False):
            agent_names = list(env.agents)
            scores = env.internal_state.get("final_scores", {})
            reason = env.internal_state.get("end_reason", "Game over.")
            response = [("environment", (("terminated", True), reason))]
            for idx, name in enumerate(agent_names):
                key = f"agent_{idx + 1}"
                score = scores.get(name, 0.0)
                response.append((key, (("complete_rating", score), "")))
            return response

        return [("environment", (("terminated", False), ""))]


# ============================================================================
# Action Handler
# ============================================================================


class ResistanceActionHandler(ActionHandler):
    """Handles actions for The Resistance."""

    def handle_action(
        self, env: SocialDeductionGame, agent_name: str, action: AgentAction
    ) -> None:
        if not isinstance(env, ResistanceEnv):
            return
        if action.action_type not in ["action", "speak"]:
            return

        if env.current_state == "Mission_proposal":
            # Only the leader proposes
            leader_idx = env.internal_state.get("leader_idx", 0)
            agent_list = list(env.agents)
            leader_name = agent_list[leader_idx % len(agent_list)]

            if agent_name != leader_name:
                return

            # Parse "propose NAME1 NAME2 ..."
            arg = action.argument.strip()
            proposed = []
            words = arg.split()
            for w in words:
                for name in env.agents:
                    if w.lower() == name.lower() and name not in proposed:
                        proposed.append(name)
                        break

            mission_num = env.internal_state.get("current_mission", 0)
            mission_sizes = env._config.get("mission_sizes", [2, 3, 2, 3, 3])
            required_size = (
                mission_sizes[mission_num] if mission_num < len(mission_sizes) else 2
            )

            if len(proposed) != required_size:
                # Default-and-continue: trim or pad to the required size so the
                # round advances. Padding picks alive players in roster order
                # who aren't already on the team.
                if len(proposed) > required_size:
                    proposed = proposed[:required_size]
                else:
                    for n in env.agents:
                        if len(proposed) >= required_size:
                            break
                        if n not in proposed and env.agent_alive.get(n, True):
                            proposed.append(n)
                env.recv_message(
                    "Environment",
                    SimpleMessage(
                        message=f"[Game] {agent_name} proposed an invalid team size; "
                        f"defaulted to: {', '.join(proposed)}."
                    ),
                )

            env.internal_state["mission_team"] = proposed
            env.recv_message(
                "Environment",
                SimpleMessage(
                    message=f"[Game] {agent_name} (Leader) proposes: {', '.join(proposed)} "
                    f"for Mission {mission_num + 1} (needs {required_size} members)."
                ),
            )

        elif env.current_state == "Mission_vote":
            arg = action.argument.strip().lower()
            if "votes" not in env.internal_state:
                env.internal_state["votes"] = {}

            if "approve" in arg or "yes" in arg:
                env.internal_state["votes"][agent_name] = "approve"
            elif "reject" in arg or "no" in arg:
                env.internal_state["votes"][agent_name] = "reject"

        elif env.current_state == "Mission_execute":
            # Only team members can act
            team = env.internal_state.get("mission_team", [])
            if agent_name not in team:
                return

            arg = action.argument.strip().lower()
            if "mission_actions" not in env.internal_state:
                env.internal_state["mission_actions"] = {}

            role = env.agent_to_role.get(agent_name, "")
            if "fail" in arg and role == "Spy":
                env.internal_state["mission_actions"][agent_name] = "fail"
            else:
                # Resistance MUST succeed; Spies default to succeed if ambiguous
                env.internal_state["mission_actions"][agent_name] = "succeed"

    def get_action_instruction(self, env: SocialDeductionGame, agent_name: str) -> str:
        if not isinstance(env, ResistanceEnv):
            return ""

        successes = env.internal_state.get("successes", 0)
        failures = env.internal_state.get("failures", 0)
        rejections = env.internal_state.get("consecutive_rejections", 0)
        mission_num = env.internal_state.get("current_mission", 0)
        mission_sizes = env._config.get("mission_sizes", [2, 3, 2, 3, 3])
        required_size = (
            mission_sizes[mission_num] if mission_num < len(mission_sizes) else 2
        )

        agent_list = list(env.agents)
        leader_idx = env.internal_state.get("leader_idx", 0)
        leader_name = agent_list[leader_idx % len(agent_list)]

        status = f"Score: Resistance {successes} - {failures} Spies. Rejections: {rejections}/5."

        if env.current_state == "Discussion":
            return (
                f"Discussion phase. Mission {mission_num + 1} (needs {required_size} members). "
                f"{status} Leader: {leader_name}. "
                f"Discuss who should go on the mission."
            )
        elif env.current_state == "Mission_proposal":
            if agent_name == leader_name:
                player_list = ", ".join(agent_list)
                return (
                    f"You are the Leader. Propose {required_size} players for Mission {mission_num + 1}. "
                    f"{status} Players: [{player_list}]. "
                    f"Use 'propose NAME1 NAME2 ...' (e.g., 'propose Alice Bob')."
                )
            else:
                return f"Waiting for {leader_name} to propose a team of {required_size}. {status}"
        elif env.current_state == "Mission_vote":
            team = env.internal_state.get("mission_team", [])
            return (
                f"Vote on the proposed team: [{', '.join(team)}]. {status} "
                f"Use 'approve' or 'reject'."
            )
        elif env.current_state == "Mission_execute":
            team = env.internal_state.get("mission_team", [])
            if agent_name in team:
                role = env.agent_to_role.get(agent_name, "")
                if role == "Spy":
                    return (
                        f"You are on the mission team. As a Spy, choose 'succeed' or 'fail'. "
                        f"{status}"
                    )
                else:
                    return f"You are on the mission team. As Resistance, you must play 'succeed'. {status}"
            else:
                return (
                    f"You are not on this mission team. Waiting for results. {status}"
                )
        return ""


# ============================================================================
# Environment
# ============================================================================


class ResistanceEnv(SocialDeductionGame):
    """The Resistance game with proposals, votes, and missions."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(action_handler=ResistanceActionHandler(), **kwargs)

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
            "leader_idx": 0,
            "current_mission": 0,
            "successes": 0,
            "failures": 0,
            "consecutive_rejections": 0,
            "mission_team": [],
            "votes": {},
            "mission_actions": {},
            "game_over": False,
            "final_scores": {},
            "end_reason": "",
        }

        # Privately tell Spies who their partners are
        spy_names = [
            name for name in self.agents if self.agent_to_role.get(name, "") == "Spy"
        ]
        for name in spy_names:
            partners = [s for s in spy_names if s != name]
            self.recv_message(
                "Environment",
                SimpleMessage(
                    message=f"[Private] You are a Spy. Your fellow Spy: {', '.join(partners)}."
                ),
                receivers=[name],
            )

        # Announce first leader
        agent_list = list(self.agents)
        leader = agent_list[0]
        self.recv_message(
            "Environment",
            SimpleMessage(message=f"[Game] {leader} is the first mission leader."),
        )

        return obs

    def _check_eliminations(self) -> None:
        """Handle mission resolution and state transitions."""
        if not self._should_transition_state():
            return

        if self.current_state == "Mission_vote":
            votes = self.internal_state.get("votes", {})
            if votes:
                approve_count = sum(1 for v in votes.values() if v == "approve")
                reject_count = sum(1 for v in votes.values() if v == "reject")

                # Announce vote results
                self.recv_message(
                    "Environment",
                    SimpleMessage(
                        message=f"[Game] Vote results: {approve_count} approve, {reject_count} reject. "
                        f"(Votes: {dict(votes)})"
                    ),
                )

                if approve_count > reject_count:
                    # Mission approved
                    self.internal_state["consecutive_rejections"] = 0
                    self.recv_message(
                        "Environment",
                        SimpleMessage(
                            message="[Game] Mission team APPROVED! Proceeding to mission execution."
                        ),
                    )
                else:
                    # Mission rejected
                    self.internal_state["consecutive_rejections"] = (
                        self.internal_state.get("consecutive_rejections", 0) + 1
                    )
                    rejections = self.internal_state["consecutive_rejections"]
                    self.recv_message(
                        "Environment",
                        SimpleMessage(
                            message=f"[Game] Mission team REJECTED! "
                            f"({rejections}/5 consecutive rejections)"
                        ),
                    )

                    # Check if 5 rejections = Spies win
                    if rejections >= 5:
                        self._end_game(
                            "Spies", "5 consecutive proposals rejected! Spies win!"
                        )
                        return

                    # Rotate leader and go back to Discussion
                    self.internal_state["leader_idx"] = (
                        self.internal_state.get("leader_idx", 0) + 1
                    )
                    agent_list = list(self.agents)
                    new_leader = agent_list[
                        self.internal_state["leader_idx"] % len(agent_list)
                    ]
                    self.recv_message(
                        "Environment",
                        SimpleMessage(message=f"[Game] New leader: {new_leader}."),
                    )

                    # Override state back to Discussion
                    self.current_state = "Discussion"
                    if hasattr(self, "_state_turn_count"):
                        self._state_turn_count["Discussion"] = 0
                    if hasattr(self, "_round_robin_idx"):
                        self._round_robin_idx = 0

                self.internal_state["votes"] = {}

        elif self.current_state == "Mission_execute":
            actions = self.internal_state.get("mission_actions", {})
            team = self.internal_state.get("mission_team", [])

            if len(actions) >= len(team):
                fail_count = sum(1 for v in actions.values() if v == "fail")
                mission_num = self.internal_state.get("current_mission", 0)

                self.recv_message(
                    "Environment",
                    SimpleMessage(
                        message=f"[Game] Mission {mission_num + 1} results: "
                        f"{fail_count} FAIL card(s) played."
                    ),
                )

                if fail_count > 0:
                    self.internal_state["failures"] = (
                        self.internal_state.get("failures", 0) + 1
                    )
                    self.recv_message(
                        "Environment",
                        SimpleMessage(
                            message=f"[Game] Mission {mission_num + 1} FAILED!"
                        ),
                    )
                else:
                    self.internal_state["successes"] = (
                        self.internal_state.get("successes", 0) + 1
                    )
                    self.recv_message(
                        "Environment",
                        SimpleMessage(
                            message=f"[Game] Mission {mission_num + 1} SUCCEEDED!"
                        ),
                    )

                successes = self.internal_state["successes"]
                failures = self.internal_state["failures"]

                # Check win conditions
                if successes >= 3:
                    self._end_game(
                        "Resistance",
                        f"3 missions succeeded! Resistance wins! (Score: {successes}-{failures})",
                    )
                    return
                elif failures >= 3:
                    self._end_game(
                        "Spies",
                        f"3 missions failed! Spies win! (Score: {successes}-{failures})",
                    )
                    return

                # Advance to next mission
                self.internal_state["current_mission"] = mission_num + 1
                self.internal_state["mission_team"] = []
                self.internal_state["mission_actions"] = {}

                # Rotate leader
                self.internal_state["leader_idx"] = (
                    self.internal_state.get("leader_idx", 0) + 1
                )
                agent_list = list(self.agents)
                new_leader = agent_list[
                    self.internal_state["leader_idx"] % len(agent_list)
                ]
                self.recv_message(
                    "Environment",
                    SimpleMessage(
                        message=f"[Game] Score: Resistance {successes} - {failures} Spies. "
                        f"New leader: {new_leader}."
                    ),
                )

    def _end_game(self, winning_team: str, reason: str) -> None:
        """Set game over with scores."""
        winners = [
            n
            for n in self.agents
            if self.role_to_team.get(self.agent_to_role.get(n, ""), "") == winning_team
        ]
        losers = [
            n
            for n in self.agents
            if self.role_to_team.get(self.agent_to_role.get(n, ""), "") != winning_team
        ]
        n_winners = len(winners)
        n_losers = len(losers)
        scores = {}
        for name in winners:
            scores[name] = 1.0 / n_winners
        for name in losers:
            scores[name] = -1.0 / n_losers if n_losers > 0 else 0.0

        self.internal_state["game_over"] = True
        self.internal_state["final_scores"] = scores
        self.internal_state["end_reason"] = reason
        self.recv_message("Environment", SimpleMessage(message=f"[Game] {reason}"))

    def _update_action_mask(self) -> None:
        """Override to handle leader-only proposal and team-only execution."""
        super()._update_action_mask()

        if self.current_state == "Mission_proposal":
            # Only the leader can act
            leader_idx = self.internal_state.get("leader_idx", 0)
            agent_list = list(self.agents)
            leader_name = agent_list[leader_idx % len(agent_list)]
            self.action_mask = [False] * len(self.agents)
            try:
                idx = agent_list.index(leader_name)
                self.action_mask[idx] = True
            except ValueError:
                pass

        elif self.current_state == "Mission_execute":
            # Only mission team members can act
            team = self.internal_state.get("mission_team", [])
            agent_list = list(self.agents)
            self.action_mask = [False] * len(self.agents)
            for name in team:
                try:
                    idx = agent_list.index(name)
                    self.action_mask[idx] = True
                except ValueError:
                    pass


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
) -> ResistanceEnv:
    return ResistanceEnv(
        env_profile=env_profile,
        config=config,
        model_name=model_name,
        evaluators=[ResistanceEvaluator(max_turn_number=100)],
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
    spy_names = [
        entry.get("name", "")
        for entry in config.get("agents", [])
        if entry.get("role", "") == "Spy"
    ]

    for idx, profile in enumerate(agent_profiles):
        agent_name = f"{profile.first_name}{' ' + profile.last_name if profile.last_name else ''}"
        role_goal = env_profile.agent_goals[idx]
        role = config.get("agents", [])[idx].get("role", "")
        secrets = config.get("role_secrets", {}).get(role, "")

        # Add spy partner info
        if role == "Spy":
            partners = [s for s in spy_names if s != agent_name]
            secrets = f"{secrets} Your fellow Spy: {', '.join(partners)}."

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
    scenario = config.get("description", "The Resistance").format(
        agent_names=", ".join(agent_names)
    )
    env_profile = EnvironmentProfile(
        scenario=scenario,
        relationship=RelationshipType.acquaintance,
        agent_goals=agent_goals,
        tag="resistance",
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

    parser = argparse.ArgumentParser(description="Run The Resistance game.")
    parser.add_argument("--config", type=str, default=str(CONFIG_PATH))
    parser.add_argument("--roster", type=str, default=str(BASE_DIR / "roster.json"))
    args = parser.parse_args()

    config = load_config(args.config)
    roster = load_config(args.roster)
    config["agents"] = roster.get("agents", [])

    agent_model_name = get_model_names(config)
    env, agents = prepare_scenario("gpt-4o", agent_model_name, config)

    print("The Resistance")
    print("=" * 60)
    print_roster(config)
    print("=" * 60)

    await arun_one_episode(
        env=env,
        agent_list=agents,
        omniscient=False,
        script_like=False,
        json_in_script=False,
        tag="test_resistance",
        push_to_db=True,
    )


if __name__ == "__main__":
    LOG_FILE = BASE_DIR / "resistance_game_debug.log"
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
