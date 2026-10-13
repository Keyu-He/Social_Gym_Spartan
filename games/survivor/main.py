"""Launcher for the Survivor game."""

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

# Turn cap. It only guards against a game that never ends (normal play ends long before it), so the models are
# not told; at the cap, players still in the game score 0 and eliminated players -1 (SurvivorEvaluator).
MAX_TURNS = 80


# ============================================================================
# Ballot reading
# ============================================================================

# The voter's own vote verb ("I vote", "my vote", "I'm voting"); "voted" and "votes" usually report
# someone else's ballot ("Carol voted for Bob", "Bob has 3 votes") and are not used as the anchor.
_VOTE_WORD = re.compile(r"\b(?:vote|voting)\b", re.I)
_SENTENCE_END = re.compile(r"[.!?](?=\s|$)")


def _mentions(text: str, names: List[str], flags: int) -> List[Tuple[int, str]]:
    """(position, name) of every whole-word mention; punctuation or markdown around a name is fine."""
    found = []
    for name in names:
        for m in re.finditer(rf"(?<![A-Za-z]){re.escape(name)}(?![A-Za-z])", text, flags):
            found.append((m.start(), name))
    return sorted(found)


def has_vote_word(text: str) -> bool:
    return bool(_VOTE_WORD.search(text))


def read_ballot(text: str, names: List[str]) -> str | None:
    """The player a ballot names, or None.

    1. The first of `names` mentioned after a 'vote'/'voting' and before the end of that sentence,
       trying each 'vote' in turn ("I vote Carol, not Dave" -> Carol; "vote: **Dave**." -> Dave;
       "...the jury vote matters. ... I vote Bob" -> Bob).
    2. Otherwise the first name anywhere after the first 'vote' ("My vote is clear. Bob has to go.").
       A ballot whose 'vote' words are followed by no name names nobody ("I'm not voting anyone out").
    3. A ballot without the word 'vote' counts for the first name in it.
    Capitalised mentions are tried before lowercase ones, so a common word ("mark", "will") does not
    shadow a name. `names` includes the voter, so a self-vote is read as one (and rejected by the
    caller) instead of being skipped in favour of a later name.
    """
    anchors = [m.end() for m in _VOTE_WORD.finditer(text)]

    def first(found: List[Tuple[int, str]], lo: int, hi: int) -> str | None:
        return next((name for pos, name in found if lo <= pos < hi), None)

    for flags in (0, re.I):
        found = _mentions(text, names, flags)
        for anchor in anchors:
            end = _SENTENCE_END.search(text, anchor)
            got = first(found, anchor, end.start() if end else len(text))
            if got:
                return got
    if anchors:
        for flags in (0, re.I):
            got = first(_mentions(text, names, flags), anchors[0], len(text))
            if got:
                return got
        return None
    for flags in (0, re.I):
        found = _mentions(text, names, flags)
        if found:
            return found[0][1]
    return None


def ballot_rule(players: List[str]) -> str:
    """How read_ballot reads a ballot, in the words the Vote and Jury_vote instructions use."""
    return (
        f"How a ballot is read: it names the first player of the game ({', '.join(players)}; this includes "
        f"you and eliminated players) written as a whole word after the word 'vote' or 'voting' in the same "
        f"sentence (a sentence ends at '.', '!' or '?'), trying each 'vote' in turn; failing that, the first "
        f"such player written anywhere after the first 'vote'; if no player is written after any 'vote', it "
        f"names nobody. A ballot without the word 'vote' or 'voting' names the first player written in it. At "
        f"each step a name written with its capital letter is tried before a lowercase one."
    )


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
            # Nobody has won: players still in the game score 0 and eliminated players keep their -1.
            # The models are not told this cap (see MAX_TURNS). The jury list, not agent_alive, says
            # who is out (agent_alive is swapped for the jury vote).
            reason = (
                f"Turn limit ({self.max_turn_number}) reached before the jury voted: "
                f"players still in the game score 0, eliminated players -1."
            )
            response = [("environment", (("terminated", True), reason))]
            if env:
                jury = set(env.internal_state.get("jury", []))
                for idx, name in enumerate(env.agents):
                    score = -1.0 if name in jury else 0.0
                    response.append(
                        (f"agent_{idx + 1}", (("complete_rating", score), ""))
                    )
            return response
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

        def notice(text: str) -> None:
            env.recv_message(
                "Environment", SimpleMessage(message=text), receivers=[agent_name]
            )

        # A ballot is read among all players, then checked: naming yourself, an eliminated player or
        # (in the jury vote) a non-finalist makes it void, with a notice, instead of letting a later
        # name in the text count. The instructions state this reading word for word (ballot_rule);
        # re-reading the 1,620 logged ballots, it changes none of them, while reading only among the
        # listed players would change one or two (results/rule_audit/fixes3/core.md, item 1).
        where = " after the word 'vote'" if has_vote_word(action.argument) else ""
        if env.current_state == "Vote":
            alive = [n for n in env.agents if env.agent_alive.get(n, False)]
            target = read_ballot(action.argument, list(env.agents))
            options = ", ".join(n for n in alive if n != agent_name)
            if target is None:
                notice(
                    f"[Game] {agent_name}: your ballot names no player still in the game{where}, "
                    f"so it does not count. You could vote for: {options}."
                )
            elif target == agent_name:
                notice(
                    f"[Game] {agent_name}: your ballot names yourself; you cannot vote for "
                    f"yourself, so it does not count."
                )
            elif target not in alive:
                notice(
                    f"[Game] {agent_name}: your ballot names {target}, who is no longer in the "
                    f"game, so it does not count. You could vote for: {options}."
                )
            else:
                env.internal_state.setdefault("votes", {})[agent_name] = target

        elif env.current_state == "Jury_vote":
            finalists = list(env.internal_state.get("finalists", []))
            target = read_ballot(action.argument, list(env.agents))
            if target is None:
                notice(
                    f"[Game] {agent_name}: your ballot names no finalist{where}, so it does not "
                    f"count. Finalists: {', '.join(finalists)}."
                )
            elif target not in finalists:
                notice(
                    f"[Game] {agent_name}: your ballot names {target}, who is not a finalist, so "
                    f"it does not count. Finalists: {', '.join(finalists)}."
                )
            else:
                env.internal_state.setdefault("jury_votes", {})[agent_name] = target

    # Kept for scripts that call the old helpers; both read ballots with read_ballot.
    def _parse_vote(
        self,
        env: SocialDeductionGame,
        voter: str,
        argument: str,
        exclude_self: bool = False,
    ) -> str | None:
        target = read_ballot(argument, list(env.agents))
        if target is None or not env.agent_alive.get(target, False):
            return None
        return None if exclude_self and target == voter else target

    def _parse_vote_from_set(self, argument: str, valid_names: set[str]) -> str | None:
        target = read_ballot(argument, sorted(valid_names))
        return target if target in valid_names else None

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
                f"Discuss who to vote out. Each player still in the game speaks once this round, in seat "
                f"order, and then everyone votes. Remember: eliminated players become jurors "
                f"who pick the winner!"
            )

        elif env.current_state == "Vote":
            alive = [n for n in env.agents if env.agent_alive.get(n, False)]
            others = [n for n in alive if n != agent_name]
            return (
                f"Vote to eliminate someone. Use 'vote NAME'. "
                f"Options: {', '.join(others)}. You cannot vote for yourself. "
                f"{ballot_rule(list(env.agents))} "
                f"If the player your ballot names is you or someone no longer in the game, or it names "
                f"nobody, the ballot does not count and you are told privately. "
                f"The player with the most votes is eliminated; a tie for the most votes is settled "
                f"by a random draw among the tied players."
            )

        elif env.current_state == "Jury_plea":
            finalists = env.internal_state.get("finalists", [])
            jury = env.internal_state.get("jury", [])
            if agent_name in finalists:
                return (
                    f"You are a finalist! Make your case to the jury "
                    f"({', '.join(jury)}) for why you should win. Each finalist makes one plea, in seat "
                    f"order, and then the jury votes. "
                    f"Other finalists: {', '.join(f for f in finalists if f != agent_name)}."
                )
            return "Waiting for finalists to make their pleas."

        elif env.current_state == "Jury_vote":
            finalists = env.internal_state.get("finalists", [])
            jury = env.internal_state.get("jury", [])
            if agent_name in jury:
                return (
                    f"As a juror, vote for who should win. "
                    f"Finalists: {', '.join(finalists)}. Use 'vote NAME'. "
                    f"{ballot_rule(list(env.agents))} "
                    f"If the player your ballot names is not a finalist, or it names nobody, the ballot "
                    f"does not count and you are told privately. The finalist with the most "
                    f"votes wins; finalists tied for the most votes (all of them, if no ballot "
                    f"counts) share the win equally."
                )
            return "Waiting for jury to vote."

        return ""


# ============================================================================
# Environment
# ============================================================================


class SurvivorEnv(SocialDeductionGame):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(action_handler=SurvivorActionHandler(), **kwargs)

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
                # Every ballot was unreadable, a self-vote or missing: nobody is eliminated, and
                # everyone is told so (each voter whose ballot did not count got a private notice).
                round_num = self.internal_state["round"] + 1
                self.internal_state["round"] = round_num
                self.recv_message(
                    "Environment",
                    SimpleMessage(
                        message=f"[Round {round_num}] No ballot counted, so nobody is eliminated."
                    ),
                )
                return

            vote_counts = Counter(votes.values())
            max_votes = max(vote_counts.values())
            tied = [n for n, c in vote_counts.items() if c == max_votes]
            eliminated = tied[0]
            if len(tied) > 1:
                # A draw among the tied players. (The show instead re-votes; if still tied, the
                # tied players are safe and the other players draw rocks.)
                eliminated = self.break_tie(tied)
                self.recv_message(
                    "Environment",
                    SimpleMessage(
                        message=f"[Game] Vote tied between {', '.join(tied)} "
                        f"({max_votes} each). Random draw: {eliminated}."
                    ),
                )

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
                    # The no-discussion variant has no pleas: the jury votes right away.
                    pleas = self.game_variant != NO_DISCUSSION

                    self.recv_message(
                        "Environment",
                        SimpleMessage(
                            message=f"[Game] Final {len(alive)} remain: {', '.join(alive)}! "
                            f"Jury: {', '.join(self.internal_state['jury'])}."
                            + (" Time for final pleas to the jury!" if pleas else "")
                        ),
                    )

                    if not pleas:
                        self._open_jury_vote()
                        return

                    # Override state to Jury_plea
                    self.current_state = "Jury_plea"
                    if hasattr(self, "_state_turn_count"):
                        self._state_turn_count["Jury_plea"] = 0
                    if hasattr(self, "_round_robin_idx"):
                        self._round_robin_idx = 0

        elif self.current_state == "Jury_plea":
            # Every finalist has pleaded.
            self._open_jury_vote()

        elif self.current_state == "Jury_vote":
            jury_votes = self.internal_state.get("jury_votes", {})
            finalists = self.internal_state.get("finalists", [])
            jury = self.internal_state.get("jury", [])

            # The jury vote is the actual win condition. Reward shape (stated in the rules text):
            #   jury-vote winner(s): +1.0 (split if tied)
            #   other finalists (made the final but didn't win the vote): 0.0
            #   jurors/eliminated: -1.0
            # The vote is decided in this one step. If no juror's ballot counted, every finalist
            # has zero votes, so all finalists are tied and share the win (no silent re-vote loop).
            vote_counts = Counter(jury_votes.values())
            top_count = vote_counts.most_common(1)[0][1] if vote_counts else 0
            winners = [n for n, c in vote_counts.items() if c == top_count and c > 0]
            if not winners:
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

            if vote_counts:
                vote_str = ", ".join(
                    f"{v}: {c} votes" for v, c in vote_counts.most_common()
                )
            else:
                vote_str = "no ballot counted, so the finalists tie with 0 votes each"
            if len(winners) == 1:
                reason = f"Jury votes: {vote_str}. {winners[0]} wins Survivor!"
            else:
                reason = (
                    f"Jury votes: {vote_str}. {' & '.join(winners)} tie and share the win!"
                )

            self.internal_state["game_over"] = True
            self.internal_state["final_scores"] = final_scores
            self.internal_state["end_reason"] = reason
            self.recv_message("Environment", SimpleMessage(message=f"[Game] {reason}"))

    def _open_jury_vote(self) -> None:
        """Swap alive status (finalists -> dead, jurors -> alive) so the engine lets only jurors act, and enter
        Jury_vote here: the config transition would recount the acting players after the swap, which gives a juror
        a plea turn whenever the jury is larger than the final (seven or more players)."""
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
        self.current_state = "Jury_vote"
        if hasattr(self, "_state_turn_count"):
            self._state_turn_count["Jury_vote"] = 0
        if hasattr(self, "_round_robin_idx"):
            self._round_robin_idx = 0
        self.recv_message(
            "Environment", SimpleMessage(message="[Game] Entering state: Jury_vote")
        )


# ============================================================================
# No-discussion variant (communication control, results/sec4_comm_control/README.md)
# ============================================================================

# SOTOPIA_GAME_VARIANT=no_discussion, read when the environment is created, removes the two speech states: each
# round is a vote (Discussion goes), and the vote that leaves three players opens the jury vote at once (Jury_plea
# goes; the line announcing the final no longer announces pleas). Ballots, ties and their seeded draws, the jury,
# the scoring, the turn cap and the other announcements stay as they are. The rules text loses the sentences about
# the speeches and says in one sentence that there is no discussion phase. Unset or empty, the game is the
# published one.
GAME_VARIANT_ENV = "SOTOPIA_GAME_VARIANT"
NO_DISCUSSION = "no_discussion"
# (published rules text, no-discussion rules text); each published text occurs once in config.json's description.
NO_DISCUSSION_RULES = [
    (
        "Each round, every player still in the game speaks once, in seat order, and then everyone votes someone "
        "out. No hidden roles or secrets.",
        "Each round, everyone still in the game votes someone out. There is no discussion phase: each round goes "
        "straight to the vote, and the finalists make no pleas before the jury votes. No hidden roles or secrets.",
    ),
    (
        "pick the final winner from the last 3 survivors, after each finalist makes one plea to the jury, in seat "
        "order.",
        "pick the final winner from the last 3 survivors.",
    ),
]


def game_variant() -> str:
    """The variant SOTOPIA_GAME_VARIANT names: '' (the published game) or 'no_discussion'."""
    variant = os.environ.get(GAME_VARIANT_ENV, "").strip()
    if variant not in ("", NO_DISCUSSION):
        raise ValueError(
            f"{GAME_VARIANT_ENV}={variant!r}: Survivor has no such variant (only {NO_DISCUSSION!r})"
        )
    return variant


def apply_variant(config: Dict[str, Any]) -> Dict[str, Any]:
    """`config` itself when no variant is set (or `config` is already the variant's). Otherwise a copy without
    Discussion and Jury_plea (the game opens with Vote, and Vote leads to Vote), with the no-discussion rules text,
    marked config["variant"]."""
    variant = game_variant()
    if not variant or config.get("variant") == variant:
        return config
    config = copy.deepcopy(config)
    text = config["description"]
    for published, replacement in NO_DISCUSSION_RULES:
        if text.count(published) != 1:
            raise ValueError(
                f"{variant}: the Survivor rules text no longer has {published!r} exactly once"
            )
        text = text.replace(published, replacement)
    config["description"] = text
    for state in ("Discussion", "Jury_plea"):
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
) -> SurvivorEnv:
    """Create the Survivor environment (the variant SOTOPIA_GAME_VARIANT names, if any)."""
    config = apply_variant(config)
    return SurvivorEnv(
        env_profile=env_profile,
        config=config,
        model_name=model_name,
        evaluators=[SurvivorEvaluator(max_turn_number=MAX_TURNS)],
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
    variant = getattr(env, "game_variant", "")
    await arun_one_episode(
        env=env,
        agent_list=agents,
        omniscient=False,
        script_like=False,
        json_in_script=False,
        tag="test_survivor",
        push_to_db=True,
        metadata={"variant": variant} if variant else None,
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
