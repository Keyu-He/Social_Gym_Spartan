"""Launcher for The Resistance social game scenario."""

from __future__ import annotations

import asyncio
import copy
import os
import logging
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


class ResistanceEvaluator(SocialGameEndEvaluator):
    """Evaluator for The Resistance win conditions."""

    def __call__(
        self, turn_number: int, messages: List[Tuple[str, Message]], **kwargs: Any
    ) -> List[Tuple[str, Tuple[Tuple[str, int | float | bool], str]]]:
        env = kwargs.get("env")
        # A game decided on the capped turn keeps its result (it used to be logged as a 0-0 timeout).
        if turn_number >= self.max_turn_number and not (
            env and env.internal_state.get("game_over", False)
        ):
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
# Reading replies
# ============================================================================

_QUOTES = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"'})
# A word before a keyword that turns it around: "I do not approve", "I won't play fail", "never fail".
_NEGATION = r"(?:\bnot|\bnever|n't|\bcannot|\bno\s+way\s+(?:i|we)(?:'ll|\s+will)?)(?:\s+\w+){0,2}\s+$"


def _first_keyword(text: str, groups: Dict[str, str]) -> str | None:
    """The group of the first keyword in the text, a negated keyword counting for the other group (two groups)."""
    low = (text or "").translate(_QUOTES).lower()
    hits = []
    for group, pattern in groups.items():
        for m in re.finditer(rf"\b(?:{pattern})\b", low):
            negated = bool(re.search(_NEGATION, low[: m.start()]))
            hits.append((m.start(), group, negated))
    if not hits:
        return None
    _, group, negated = min(hits)
    if negated:
        return next(g for g in groups if g != group)
    return group


def read_team_vote(text: str) -> str | None:
    """'approve', 'reject', or None when the reply is neither.

    The first approve/reject word decides, a negated one counting the other way ("I do not approve" rejects);
    "yes" or "no" as whole words count only when neither word is there. Votes used to be read by substring,
    approve first, so "I reject ... shouldn't be so quick to approve" approved and "another chance" rejected.
    """
    vote = _first_keyword(
        text,
        {
            "approve": r"approve[sd]?|approving|approval|accept(?:s|ed|ing)?",
            "reject": r"reject(?:s|ed|ing)?|rejection|disapprove[sd]?|veto",
        },
    )
    return vote or _first_keyword(
        text, {"approve": r"yes|yeah|aye", "reject": r"no|nope|nay"}
    )


def read_mission_card(text: str) -> str | None:
    """'succeed', 'fail', or None when the reply is neither.

    The first card word decides, a negated one counting the other way ("I will not fail" succeeds). Any spy text
    containing "fail" used to be a Fail, so "I will not fail this mission, I play succeed" failed the mission.
    """
    return _first_keyword(
        text,
        {
            "succeed": r"succeed(?:s|ed|ing)?|success(?:ful)?",
            "fail": r"fail(?:s|ed|ing|ure)?|sabotage[sd]?",
        },
    )


def read_proposal(text: str, names: List[str]) -> List[str]:
    """The players a proposal names, in the order given.

    Names match as whole words in any case, so "propose Alice, Bob." names both (a name with a comma or full stop
    attached used to be missed, and the team was then padded in seat order). After the word "propose" the team runs
    to the end of that sentence ("propose Alice and Bob because Carol ..." names Alice and Bob); a reply without
    "propose", or with no name after it, counts every player it names.
    """
    text = (text or "").translate(_QUOTES)

    def named(segment: str) -> List[str]:
        found = []
        for n in names:
            m = re.search(
                r"(?<![A-Za-z])"
                + r"\s+".join(map(re.escape, n.split()))
                + r"(?![A-Za-z])",
                segment,
                re.I,
            )
            if m:
                found.append((m.start(), n))
        return [n for _, n in sorted(found)]

    m = re.search(r"\bpropos(?:e|es|ing|al)\b", text, re.IGNORECASE)
    if m:
        segment = re.split(
            r"[;!?\n]|\.(?:\s|$)|\b(?:because|since|as|while)\b",
            text[m.end() :],
            maxsplit=1,
        )[0]
        team = named(segment)
        if team:
            return team
    return named(text)


def _acting(env: SocialDeductionGame, agent_name: str) -> bool:
    """Whether the engine asked this player to act in the step being processed."""
    try:
        return bool(env.action_mask[env.agents.index(agent_name)])
    except (ValueError, IndexError):
        return False


# ============================================================================
# Action Handler
# ============================================================================


class ResistanceActionHandler(ActionHandler):
    """Handles actions for The Resistance."""

    def handle_action(
        self, env: SocialDeductionGame, agent_name: str, action: AgentAction
    ) -> None:
        if not isinstance(env, ResistanceEnv) or not _acting(env, agent_name):
            return
        # A failed model call arrives as 'none'; it is read as an empty reply, and every phase says what that
        # means (it used to be dropped: the old team stayed, or the mission never resolved).
        text = (
            action.argument.strip() if action.action_type in ["action", "speak"] else ""
        )

        if env.current_state == "Mission_proposal":
            # Only the leader proposes
            leader_idx = env.internal_state.get("leader_idx", 0)
            agent_list = list(env.agents)
            leader_name = agent_list[leader_idx % len(agent_list)]

            if agent_name != leader_name:
                return

            proposed = read_proposal(text, agent_list)

            mission_num = env.internal_state.get("current_mission", 0)
            mission_sizes = env._config.get("mission_sizes", [2, 3, 2, 3, 3])
            required_size = (
                mission_sizes[mission_num] if mission_num < len(mission_sizes) else 2
            )

            if len(proposed) != required_size:
                if not text:
                    issue = "was missing"
                elif not proposed:
                    issue = "named no player"
                else:
                    issue = f"named {len(proposed)} ({', '.join(proposed)})"
                if not env.internal_state.get("proposal_retry"):
                    # First wrong-size proposal: announced, and the same leader proposes once more (the state
                    # stays Mission_proposal for one more turn; see ResistanceEnv._should_transition_state).
                    env.internal_state["proposal_retry"] = True
                    env.recv_message(
                        "Environment",
                        SimpleMessage(
                            message=f"[Game] {agent_name}'s proposal {issue}, but Mission {mission_num + 1} needs "
                            f"exactly {required_size} players. {agent_name} proposes once more; if that proposal "
                            f"is also the wrong size, the engine trims it or fills it up in seat order."
                        ),
                    )
                    return
                # Second wrong-size proposal: trimmed to the first players named, or filled up with players in
                # seat order who are not on it yet; announced, and the team goes to the vote. Not a rejection.
                if len(proposed) > required_size:
                    proposed = proposed[:required_size]
                    how = f"trimmed to the first {required_size} named"
                else:
                    for n in env.agents:
                        if len(proposed) >= required_size:
                            break
                        if n not in proposed and env.agent_alive.get(n, True):
                            proposed.append(n)
                    how = f"filled up in seat order to {required_size}"
                env.recv_message(
                    "Environment",
                    SimpleMessage(
                        message=f"[Game] {agent_name}'s second proposal {issue} too, so the team is {how}: "
                        f"{', '.join(proposed)}."
                    ),
                )
            env.internal_state["proposal_retry"] = False

            # A new proposal is a new team: nothing from an earlier team carries over.
            env.internal_state["mission_team"] = proposed
            env.internal_state["mission_actions"] = {}
            env.recv_message(
                "Environment",
                SimpleMessage(
                    message=f"[Game] {agent_name} (Leader) proposes: {', '.join(proposed)} "
                    f"for Mission {mission_num + 1} (needs {required_size} members)."
                ),
            )

        elif env.current_state == "Mission_vote":
            if "votes" not in env.internal_state:
                env.internal_state["votes"] = {}
            vote = read_team_vote(text)
            if vote is None:
                # Every player votes; a reply that is neither approve nor reject counts as a reject (a tie
                # rejects anyway), so approval always needs a majority of the table. Announced at the tally.
                vote = "reject"
                env.internal_state.setdefault("vote_problems", {})[agent_name] = (
                    "could not be read as approve or reject" if text else "was missing"
                )
            env.internal_state["votes"][agent_name] = vote

        elif env.current_state == "Mission_execute":
            # Only team members can act
            team = env.internal_state.get("mission_team", [])
            if agent_name not in team:
                return

            if "mission_actions" not in env.internal_state:
                env.internal_state["mission_actions"] = {}

            role = env.agent_to_role.get(agent_name, "")
            card = read_mission_card(text)
            if role != "Spy":
                # Resistance MUST succeed
                played = "succeed"
                note = (
                    ""
                    if card == "succeed"
                    else " Resistance members always play succeed."
                )
            elif card:
                played, note = card, ""
            else:
                # A Spy's reply that is neither card (or no reply) is a succeed
                played = "succeed"
                note = (
                    " Your reply was neither 'succeed' nor 'fail', so it counts as succeed."
                    if text
                    else (" You sent no card, so it counts as succeed.")
                )
            env.internal_state["mission_actions"][agent_name] = played
            # Cards are secret: the player alone is told how its card was read.
            env.recv_message(
                "Environment",
                SimpleMessage(message=f"[Private] Your mission card: {played}.{note}"),
                receivers=[agent_name],
            )

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
                if env.internal_state.get("proposal_retry"):
                    again = (
                        f"Your proposal had the wrong size. Propose exactly {required_size} players for Mission "
                        f"{mission_num + 1}. If this proposal is also the wrong size, the engine trims it to the "
                        f"first {required_size} named or fills it up in seat order."
                    )
                else:
                    again = (
                        f"Propose {required_size} players for Mission {mission_num + 1}. A proposal of the wrong "
                        f"size is announced and you propose once more."
                    )
                return (
                    f"You are the Leader. {again} "
                    f"{status} Players: [{player_list}]. "
                    f"Use 'propose NAME1 NAME2 ...' (e.g., 'propose Alice Bob')."
                )
            else:
                return f"Waiting for {leader_name} to propose a team of {required_size}. {status}"
        elif env.current_state == "Mission_vote":
            team = env.internal_state.get("mission_team", [])
            return (
                f"Vote on the proposed team: [{', '.join(team)}]. {status} "
                f"Use 'approve' or 'reject'. A reply that is neither counts as a reject."
            )
        elif env.current_state == "Mission_execute":
            team = env.internal_state.get("mission_team", [])
            if agent_name in team:
                role = env.agent_to_role.get(agent_name, "")
                if role == "Spy":
                    return (
                        f"You are on the mission team. As a Spy, choose 'succeed' or 'fail'. "
                        f"A reply that is neither counts as succeed. {status}"
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
            "leader_idx": 0,
            "current_mission": 0,
            "successes": 0,
            "failures": 0,
            "consecutive_rejections": 0,
            "proposal_retry": False,
            "mission_team": [],
            "votes": {},
            "vote_problems": {},
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

    def _should_transition_state(self) -> bool:
        # After a first wrong-size proposal the leader proposes once more before the vote.
        if self.current_state == "Mission_proposal" and self.internal_state.get(
            "proposal_retry"
        ):
            return False
        return super()._should_transition_state()

    def _check_eliminations(self) -> None:
        """Handle mission resolution and state transitions."""
        if not self._should_transition_state():
            return

        if self.current_state == "Mission_vote":
            votes = self.internal_state.get("votes", {})
            # Each vote that could not be read (or was missing) counts as a reject, and says so; a round in which
            # no vote was readable is a rejection like any other (it used to send the team out unapproved).
            for name, problem in self.internal_state.get("vote_problems", {}).items():
                self.recv_message(
                    "Environment",
                    SimpleMessage(
                        message=f"[Game] {name}'s vote {problem}; it counts as a reject."
                    ),
                )
            self.internal_state["vote_problems"] = {}
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
                # Mission rejected: the team is gone, and nothing of it carries over to the next proposal
                self.internal_state["mission_team"] = []
                self.internal_state["mission_actions"] = {}
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
                    self.internal_state["votes"] = {}
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

                # Override state back to Discussion (to the next proposal in the no-discussion variant, announced
                # as the proposal after a mission is)
                round_start = (
                    "Mission_proposal"
                    if self.game_variant == NO_DISCUSSION
                    else "Discussion"
                )
                self.current_state = round_start
                if hasattr(self, "_state_turn_count"):
                    self._state_turn_count[round_start] = 0
                if hasattr(self, "_round_robin_idx"):
                    self._round_robin_idx = 0
                if round_start != "Discussion":
                    self.recv_message(
                        "Environment",
                        SimpleMessage(message=f"[Game] Entering state: {round_start}"),
                    )

            self.internal_state["votes"] = {}

        elif self.current_state == "Mission_execute":
            actions = self.internal_state.get("mission_actions", {})
            team = self.internal_state.get("mission_team", [])

            if team and len(actions) >= len(team):
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
# No-discussion variant (communication control, results/sec4_comm_control/README.md)
# ============================================================================

# SOTOPIA_GAME_VARIANT=no_discussion, read when the environment is created, removes Discussion: the game opens with
# the first leader's proposal, and every mission or rejected team is followed by the next proposal. Roles,
# proposals, votes, missions, win rules, the turn cap and the engine's announcements stay as they are. The rules
# text loses its discussion sentence and says in one sentence that there is no discussion phase. Unset or empty,
# the game is the published one.
GAME_VARIANT_ENV = "SOTOPIA_GAME_VARIANT"
NO_DISCUSSION = "no_discussion"
# (published rules text, no-discussion rules text); each published text occurs once in config.json's description.
NO_DISCUSSION_RULES = [
    (
        "No player is ever eliminated.\n\nGAME FLOW:\n1. Discussion: All players discuss strategy publicly.\n"
        "2. Mission_proposal:",
        "No player is ever eliminated. There is no discussion phase: each new leader proposes a team right away."
        "\n\nGAME FLOW:\n1. Mission_proposal:",
    ),
    ("\n3. Mission_vote:", "\n2. Mission_vote:"),
    (
        "leader rotates, back to Discussion.",
        "leader rotates, back to Mission_proposal.",
    ),
    ("\n4. Mission_execute:", "\n3. Mission_execute:"),
]


def game_variant() -> str:
    """The variant SOTOPIA_GAME_VARIANT names: '' (the published game) or 'no_discussion'."""
    variant = os.environ.get(GAME_VARIANT_ENV, "").strip()
    if variant not in ("", NO_DISCUSSION):
        raise ValueError(
            f"{GAME_VARIANT_ENV}={variant!r}: The Resistance has no such variant (only {NO_DISCUSSION!r})"
        )
    return variant


def apply_variant(config: Dict[str, Any]) -> Dict[str, Any]:
    """`config` itself when no variant is set (or `config` is already the variant's). Otherwise a copy without
    Discussion (the game opens with Mission_proposal, and Mission_execute leads to it), with the no-discussion rules
    text, marked config["variant"]."""
    variant = game_variant()
    if not variant or config.get("variant") == variant:
        return config
    config = copy.deepcopy(config)
    text = config["description"]
    for published, replacement in NO_DISCUSSION_RULES:
        if text.count(published) != 1:
            raise ValueError(
                f"{variant}: the Resistance rules text no longer has {published!r} exactly once"
            )
        text = text.replace(published, replacement)
    config["description"] = text
    _skip_state(config, "Discussion")
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


def longest_game(mission_sizes: List[int], players: int = 5) -> int:
    """The most turns a game can take. A proposal costs one turn per player for Discussion, two proposal turns (a
    wrong-size first proposal is asked once more) and one vote; the fifth rejected proposal in a row ends the game,
    so a mission takes at most four rejected proposals plus an approved one and its one execution turn. The longest
    game plays every mission (the score is 2-2 before the last)."""
    proposal = players + 2 + 1
    return len(mission_sizes) * (5 * proposal + 1)


# A backstop only, never told to the models: every game ends by turn 205 (5 missions x (5 proposals x 8 turns + 1)),
# 180 when no proposal is asked again, so this cap never cuts a game. It used to be 100, which cut games of 14 or
# more proposals. The no-discussion variant keeps it (its longest game is 80 turns).
MAX_TURNS = longest_game([2, 3, 2, 3, 3])


def create_environment(
    env_profile: EnvironmentProfile, model_name: str, config: Dict[str, Any]
) -> ResistanceEnv:
    config = apply_variant(config)
    return ResistanceEnv(
        env_profile=env_profile,
        config=config,
        model_name=model_name,
        evaluators=[
            ResistanceEvaluator(
                max_turn_number=max(
                    MAX_TURNS,
                    longest_game(
                        config.get("mission_sizes", [2, 3, 2, 3, 3]),
                        len(config.get("agents", [])) or 5,
                    ),
                )
            )
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
    config = apply_variant(config)
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

    variant = getattr(env, "game_variant", "")
    await arun_one_episode(
        env=env,
        agent_list=agents,
        omniscient=False,
        script_like=False,
        json_in_script=False,
        tag="test_resistance",
        push_to_db=True,
        metadata={"variant": variant} if variant else None,
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
