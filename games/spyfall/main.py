"""Launcher for the Spyfall social game scenario."""

from __future__ import annotations

import asyncio
import logging
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

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
from sotopia.envs import SocialDeductionGame
from sotopia.envs.evaluators import SocialGameEndEvaluator
from sotopia.envs.social_game import (
    SOCIAL_GAME_PROMPT_TEMPLATE,
    ActionHandler,
    load_config,
)
from sotopia.messages import AgentAction, Message, Observation, SimpleMessage
from sotopia.server import arun_one_episode

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"

os.environ.setdefault("REDIS_OM_URL", "redis://:@localhost:6379")
redis.Redis(host="localhost", port=6379)

# Loggers (configured in main if running standalone)
_gen_logger = logging.getLogger("sotopia.generation")
_env_logger = logging.getLogger("sotopia.envs.social_game")


# ============================================================================
# Spyfall game-end evaluator
# ============================================================================


class SpyfallGameEndEvaluator(SocialGameEndEvaluator):
    """Evaluator that checks spyfall win conditions."""

    def _check_win_conditions(  # type: ignore[override]
        self, env: Any, turn_number: int, messages: List[tuple[str, Message]]
    ) -> tuple[bool, str, Dict[str, float]]:
        """Check if game has ended based on spyfall win conditions."""

        # The Spy stopped the game to guess the location: a correct guess wins for the Spy, a wrong one loses.
        guess = env.internal_state.get("spy_guess")
        if guess:
            spy_team = env.role_to_team.get("Spy", "Spy")
            winners = [
                a
                for a in env.agents
                if (env.role_to_team.get(env.agent_to_role.get(a, ""), "") == spy_team)
                == guess["correct"]
            ]
            losers = [a for a in env.agents if a not in winners]
            rewards = {a: 1.0 / len(winners) for a in winners}
            rewards.update({a: -1.0 / len(losers) for a in losers})
            return True, guess["message"], rewards

        non_spies_count = 0
        spy_count = 0

        for agent_name, alive in env.agent_alive.items():
            role = env.agent_to_role.get(agent_name, "")
            if role == "Spy":
                if not alive:
                    # Spy is eliminated
                    pass
                else:
                    spy_count += 1
            else:
                if alive:
                    non_spies_count += 1

        # Check end conditions from config
        end_conditions = env._config.get("end_conditions", [])
        for condition in end_conditions:
            cond_type = condition.get("type")

            if cond_type == "team_eliminated":
                team = condition.get("team", "")  # e.g. "Spy"
                # If target team is eliminated (Spy count is 0)
                if team == "Spy" and spy_count == 0:
                    winner = condition.get("winner", "")
                    msg = condition.get("message", f"{winner} wins!")
                    env.recv_message("Environment", SimpleMessage(message=msg))

                    # Calculate rewards
                    winning_team = [
                        a
                        for a in env.agents
                        if env.role_to_team.get(env.agent_to_role.get(a, ""), "")
                        == winner
                    ]
                    losing_team = [
                        a
                        for a in env.agents
                        if env.role_to_team.get(env.agent_to_role.get(a, ""), "")
                        != winner
                    ]
                    n_winners = len(winning_team)
                    n_losers = len(losing_team)
                    rewards = {}
                    for agent_name in winning_team:
                        rewards[agent_name] = 1.0 / n_winners
                    for agent_name in losing_team:
                        rewards[agent_name] = -1.0 / n_losers if n_losers > 0 else 0.0

                    return True, msg, rewards

            elif cond_type == "parity":
                # Spy wins if spy count >= non-spy count
                team1 = condition.get("team", "")
                team2 = condition.get("other", "")
                # Count current alive
                team1_count = 0
                team2_count = 0
                for agent_name, alive in env.agent_alive.items():
                    if alive:
                        role = env.agent_to_role.get(agent_name, "")
                        team = env.role_to_team.get(role, "")
                        if team == team1:
                            team1_count += 1
                        elif team == team2:
                            team2_count += 1

                if team1_count >= team2_count and team1_count > 0:
                    winner = condition.get("winner", "")
                    msg = condition.get("message", f"{winner} wins!")
                    env.recv_message("Environment", SimpleMessage(message=msg))

                    # Calculate rewards
                    winning_team = [
                        a
                        for a in env.agents
                        if env.role_to_team.get(env.agent_to_role.get(a, ""), "")
                        == winner
                    ]
                    losing_team = [
                        a
                        for a in env.agents
                        if env.role_to_team.get(env.agent_to_role.get(a, ""), "")
                        != winner
                    ]
                    n_winners = len(winning_team)
                    n_losers = len(losing_team)
                    rewards = {}
                    for agent_name in winning_team:
                        rewards[agent_name] = 1.0 / n_winners
                    for agent_name in losing_team:
                        rewards[agent_name] = -1.0 / n_losers if n_losers > 0 else 0.0
                    return True, msg, rewards

        return False, "", {}

    def __call__(
        self, turn_number: int, messages: List[tuple[str, Message]], **kwargs: Any
    ) -> List[tuple[str, tuple[tuple[str, int | float | bool], str]]]:
        # Extract environment from kwargs
        env = kwargs.get("env")
        if not env:
            if turn_number >= self.max_turn_number:
                return [("environment", (("terminated", True), "Max turns reached"))]
            return [("environment", (("terminated", False), ""))]

        # Win conditions first: a vote that decides the game on the capped turn is a win, not a time-out.
        terminated, reason, rewards = self._check_win_conditions(
            env, turn_number, messages
        )
        if not terminated and turn_number >= self.max_turn_number:
            # Time is up without a conviction: the Spy wins, as in the rulebook when the timer runs out with no
            # one convicted. It used to be a draw, every player 0.
            spy_team = env.role_to_team.get("Spy", "Spy")
            spies = [a for a in env.agents if env.agent_to_role.get(a) == "Spy"]
            terminated = True
            reason = (
                f"[Game] Time is up: {self.max_turn_number} turns have passed and the Spy was not voted out. "
                f"{', '.join(spies)} was the Spy. The Spy wins."
            )
            env.recv_message("Environment", SimpleMessage(message=reason))
            winners = [
                a
                for a in env.agents
                if env.role_to_team.get(env.agent_to_role.get(a, ""), "") == spy_team
            ]
            losers = [a for a in env.agents if a not in winners]
            rewards = {a: 1.0 / len(winners) for a in winners}
            rewards.update({a: -1.0 / len(losers) for a in losers})

        response: List[tuple[str, tuple[tuple[str, int | float | bool], str]]] = [
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
# Spyfall-specific game logic
# ============================================================================


class SpyfallActionHandler(ActionHandler):
    """Handles actions for the Spyfall game."""

    def handle_action(
        self, env: SocialDeductionGame, agent_name: str, action: AgentAction
    ) -> None:
        """Handle a single action from an agent based on current state."""

        if env.current_state == "Round_questioning":
            # A questioning turn is a statement ('speak'), or, for the Spy only, the action that stops the game to
            # guess the location (rulebook: the Spy may stop the clock at any time except during an accusation,
            # reveal the Spy card and name the location; right, the Spy wins; wrong, the others win).
            if not _acting(env, agent_name) or action.action_type != "action":
                return
            if env.agent_to_role.get(agent_name) != "Spy":
                env.recv_message(
                    "Environment",
                    SimpleMessage(
                        message=f"[Game] {agent_name} used 'action' on a questioning turn. Only the Spy can stop "
                        f"the game to guess the location, so it has no effect."
                    ),
                )
                return
            _resolve_spy_guess(env, agent_name, action.argument)

        elif env.current_state == "Round_vote":
            # Collect votes for elimination
            if "votes" not in env.internal_state:
                env.internal_state["votes"] = {}
            if not _acting(env, agent_name):
                return

            # Only a player still in the game can be voted for. Anything else is an abstention, announced to all
            # (the votes themselves are public), so the mistake stays visible and cannot cancel the vote.
            text = action.argument if action.action_type == "action" else ""
            target, mentioned = read_named_player(text, list(env.agents))
            if target and env.agent_alive.get(target, False):
                env.internal_state["votes"][agent_name] = target
                return
            if target:
                reason = f"{agent_name}'s vote for {target} does not count: {target} is already out of the game."
            elif mentioned:
                reason = f"{agent_name}'s vote names several players ({', '.join(mentioned)}) and no single 'vote NAME'."
            elif re.search(rf"\b(?:{_VOTE_VERB})\b", text, re.IGNORECASE):
                reason = f"{agent_name}'s vote names no player."
            else:
                reason = f"{agent_name} cast no vote."
            env.recv_message(
                "Environment",
                SimpleMessage(message=f"[Game] {reason} Counted as an abstention."),
            )

    def get_action_instruction(self, env: SocialDeductionGame, agent_name: str) -> str:
        """Get specific action instructions for an agent based on current state."""

        if env.current_state == "Round_vote":
            return (
                "It is voting time. You MUST use the command 'vote NAME' to vote for the player you suspect is the "
                f"Spy. e.g. 'vote Alice'. Players still in the game: {_living_list(env)}. A vote for a player who is "
                "out, or a vote that names no single player, counts as an abstention. The player with the most "
                "counted votes is out; if the top vote is tied, one of the tied players is drawn at random "
                f"(announced to everyone). {_time_left(env)}"
            )

        if env.current_state == "Round_questioning" and _acting(env, agent_name):
            if env.agent_to_role.get(agent_name) == "Spy":
                return (
                    "It is your turn. Use 'speak' to ask or answer a question, or stop the game and guess the "
                    "location with the action 'guess LOCATION', naming one of the possible locations. A correct "
                    "guess wins the game for you; a wrong guess, or one that names no single location from the "
                    f"list, loses it. {_time_left(env)}"
                )
            return f"It is your turn to ask or answer a question. {_time_left(env)}"

        return ""


def _acting(env: SocialDeductionGame, agent_name: str) -> bool:
    """Whether the engine asked this player to act in the step being processed."""
    try:
        return bool(env.action_mask[env.agents.index(agent_name)])
    except (ValueError, IndexError):
        return False


def _living_list(env: SocialDeductionGame) -> str:
    return ", ".join(a for a in env.agents if env.agent_alive.get(a, False))


# The game's time limit in turns (each questioning turn and each vote is one turn). The rulebook plays to a timer;
# when time runs out with no one convicted, the Spy wins.
MAX_TURNS = 20


def _time_left(env: SocialDeductionGame) -> str:
    cap = next(
        (
            e.max_turn_number
            for e in env.evaluators
            if isinstance(e, SpyfallGameEndEvaluator)
        ),
        MAX_TURNS,
    )
    left = max(cap - env.turn_number, 0)
    return (
        f"Time left: {left} of {cap} turns, this one included. If time runs out before the Spy is voted out, the "
        "Spy wins."
    )


_QUOTES = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"'})
# Between a command word and the name it takes: punctuation, markdown, quotes, "for" ("vote for Bob").
_GAP = r"""[\s:*_'"`@(\[-]*(?:(?:for|against|out)\s+[\s*_'"`@(\[-]*)?"""
_VOTE_VERB = r"vot(?:e|es|ed|ing)"
# A word that turns what follows around ("I won't vote Bob", "Carol, not Bob", "It's not hail").
_NEG_CUE = r"(?:\bnot|\bnever|n't|\bcannot|\brather\s+than|\binstead\s+of)\b"
# A negation reaches to the end of its clause.
_CLAUSE_BREAK = r"[.;:!?,\n]|\bbut\b"
# What may stand between a negation and the name it negates ("not Bob", "not for **Bob**").
_NAME_LEAD = r"""(?:\s+(?:for|on|against|with))?[\s*_'"`@(\[-]*$"""


def _name_re(name: str) -> str:
    return (
        r"(?<![A-Za-z])" + r"\s+".join(map(re.escape, name.split())) + r"(?![A-Za-z])"
    )


def _clause_before(text: str, pos: int) -> str:
    return re.split(_CLAUSE_BREAK, text[:pos], flags=re.IGNORECASE)[-1]


def _negated_command(text: str, pos: int, verb: str, names: List[str]) -> bool:
    """Whether the command word at ``pos`` is negated: a negation at most three words before it in its clause, with
    no command word or player name in between ("I won't vote", "I do not want to vote", "don't kill")."""
    m = re.search(
        rf"{_NEG_CUE}((?:\W+\w+){{0,3}})\W*$", _clause_before(text, pos), re.IGNORECASE
    )
    if not m:
        return False
    between = m.group(1)
    return not re.search(rf"\b(?:{verb})\b", between, re.IGNORECASE) and not any(
        re.search(_name_re(n), between) for n in names
    )


def read_named_player(
    text: str, names: List[str], verb: str = _VOTE_VERB
) -> Tuple[str | None, List[str]]:
    """The player a move names, read as a person at the table would read it.

    The name right after the command word counts ("vote Bob", "vote for Bob", "vote: **'Bob'**"), wherever it
    stands in the reply, unless the command is negated: "I won't vote Bob, vote Carol" is a vote for Carol.
    Otherwise the reply counts for a player only if it is just that name or names exactly one player (written with
    a capital, so the word "mark" is not a vote for Mark), and a negated mention does not count: "My vote goes to
    Carol, not Bob" names Carol. Names match as whole words, so punctuation, quotes, markdown and a trailing "'s" do
    not matter. Returns (player, []) or, when no single player is named, (None, the players mentioned).
    """
    text = (text or "").translate(_QUOTES)
    hits: List[Tuple[int, str]] = []
    negated: List[Tuple[int, int]] = []
    for n in names:
        for m in re.finditer(
            rf"\b(?:{verb})\b{_GAP}{_name_re(n)}", text, re.IGNORECASE
        ):
            if not _negated_command(text, m.start(), verb, names):
                hits.append((m.start(), n))
                continue
            stop = re.search(_CLAUSE_BREAK, text[m.end() :], re.IGNORECASE)
            negated.append((m.start(), m.end() + stop.start() if stop else len(text)))
    if hits:
        return min(hits)[1], []
    words = re.sub(r"[\W_]+", " ", text).strip().lower()
    for n in names:
        if words == " ".join(n.lower().split()):
            return n, []
    mentioned = [
        n
        for n in names
        if any(
            not re.search(
                _NEG_CUE + _NAME_LEAD, _clause_before(text, m.start()), re.IGNORECASE
            )
            and not any(a <= m.start() < b for a, b in negated)
            for m in re.finditer(_name_re(n), text)
        )
    ]
    if len(mentioned) == 1:
        return mentioned[0], []
    return None, mentioned


_GUESS_LEAD = (
    r"^\s*(?:i\s+(?:think|believe|guess|bet)|i'?m\s+guessing|i\s+am\s+guessing|my\s+(?:final\s+)?guess(?:\s+is)?"
    r"|guess(?:ing)?|the\s+location\s+is|we\s+are\s+(?:at|in|on)|it\s+is|its)\b\s*(?:that\b)?\s*[:\-]?\s*"
)


def drop_negated(text: str) -> str:
    """The text without its negated parts: each negation and the rest of its clause, so "It's not hail, it's sleet"
    becomes "It's , it's sleet" and "I don't know" becomes "I"."""
    return re.sub(
        rf"\w*{_NEG_CUE}.*?(?={_CLAUSE_BREAK}|$)", " ", text or "", flags=re.IGNORECASE
    )


def read_location(text: str, locations: List[str]) -> str | None:
    """The location a guess names, or None when it names no single location from the list.

    The guess is the first clause of the reply that says something once lead-ins ("guess", "I think we are at",
    "Guess: the") are removed; it names a location if it is that location ("guess Beach", "Guess: the space
    station.", "I guess we are at the Movie Studio"). Otherwise a reply that mentions exactly one location names it.
    A negated part does not count ("It is not the Bank. guess Casino" names Casino). Case, an article, a plural and
    joined words ("spacestation") do not matter; "Zoo or Beach" names none.
    """
    raw = drop_negated((text or "").translate(_QUOTES)).lower().replace("'", "")

    def words(t: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", t).strip()

    patterns = {
        loc: r"(?<![a-z0-9])"
        + r"\s*".join(map(re.escape, loc.lower().split()))
        + r"(?:e?s)?(?![a-z0-9])"
        for loc in locations
    }
    phrase = ""
    for clause in re.split(
        r"[.,;!?()\n]|\s[-\u2013\u2014]\s|\b(?:because|since|as|given|based)\b", raw
    ):
        phrase, previous = words(clause), None
        while previous != phrase:
            previous = phrase
            phrase = re.sub(r"^(?:a|an|the)\s+", "", re.sub(_GUESS_LEAD, "", phrase))
        if phrase:
            break
    exact = [loc for loc, p in patterns.items() if re.fullmatch(p, phrase)]
    if len(exact) == 1:
        return exact[0]
    mentioned = [loc for loc, p in patterns.items() if re.search(p, words(raw))]
    return mentioned[0] if len(mentioned) == 1 else None


def _resolve_spy_guess(env: SocialDeductionGame, spy: str, text: str) -> None:
    """The Spy stops the game and names the location; the result is announced and ends the game."""
    location = env.internal_state.get("location", "")
    guess = read_location(text, env.internal_state.get("location_list", []))
    head = f"[Game] {spy} stops the game and reveals being the Spy."
    if guess is None:
        shown = (text or "").strip().replace("\n", " ")[:80]
        message = (
            f'{head} The guess "{shown}" names no single location from the list, so it counts as a wrong guess. '
            f"The location was {location}. The Non-Spies win."
        )
    elif guess.lower() == location.lower():
        message = f"{head} Guess: {guess}. Correct! The Spy wins."
    else:
        message = f"{head} Guess: {guess}. Wrong: the location was {location}. The Non-Spies win."
    env.internal_state["spy_guess"] = {
        "spy": spy,
        "guess": guess,
        "correct": guess is not None and guess.lower() == location.lower(),
        "message": message,
    }
    env.recv_message("Environment", SimpleMessage(message=message))


class SpyfallEnv(SocialDeductionGame):
    """Spyfall game with questioning and voting."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(action_handler=SpyfallActionHandler(), **kwargs)

    def reset(
        self,
        seed: int | None = None,
        options: Dict[str, str] | None = None,
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
        self.internal_state["location"] = _location_of(self._config)
        self.internal_state["location_list"] = sorted(
            self._config.get("location_pool", [])
        )
        self.internal_state["spy_guess"] = None
        if self.action_handler:  # the opener's instruction (the Spy's guess option) depends on the state above
            for name in self.agents:
                obs[name].available_actions = self._get_available_actions(name)
                obs[
                    name
                ].action_instruction = self.action_handler.get_action_instruction(
                    self, name
                )
        return obs

    def _get_available_actions(
        self, agent_name: str
    ) -> List[Literal["none", "speak", "non-verbal communication", "action", "leave"]]:
        """On a questioning turn the Spy may also 'action' (guess the location); everyone else may only speak."""
        actions = super()._get_available_actions(agent_name)
        if (
            self.current_state == "Round_questioning"
            and actions == ["speak"]
            and self.agent_to_role.get(agent_name) == "Spy"
        ):
            return ["speak", "action"]
        return actions

    def _should_transition_state(self) -> bool:
        if self.internal_state.get("spy_guess"):
            return False  # the guess ended the game
        return super()._should_transition_state()

    def _check_eliminations(self) -> None:
        """Apply eliminations based on collected actions."""
        # Only apply eliminations if we are about to transition state
        if not self._should_transition_state():
            return

        if self.current_state == "Round_vote":
            # Tally votes and eliminate most voted player (every recorded vote names a living player)
            votes = self.internal_state.get("votes", {})
            if not votes:
                self.recv_message(
                    "Environment",
                    SimpleMessage(
                        message="[Game] No valid votes: nobody is voted out this round."
                    ),
                )
            else:
                vote_counts: Dict[str, int] = {}
                for target in votes.values():
                    vote_counts[target] = vote_counts.get(target, 0) + 1

                if vote_counts:
                    # How each vote was read, so a misread is visible.
                    self.recv_message(
                        "Environment",
                        SimpleMessage(
                            message="[Game] Votes counted: "
                            + ", ".join(
                                f"{v} -> {votes[v]}" for v in self.agents if v in votes
                            )
                            + "."
                        ),
                    )
                    top = max(vote_counts.values())
                    tied = [n for n, c in vote_counts.items() if c == top]
                    eliminated = tied[0]
                    if len(tied) > 1:
                        eliminated = self.break_tie(tied)
                        self.recv_message(
                            "Environment",
                            SimpleMessage(
                                message=f"[Game] Vote tied between {', '.join(tied)} "
                                f"({top} each). Random draw: {eliminated}."
                            ),
                        )
                    self.agent_alive[eliminated] = False
                    self.recv_message(
                        "Environment",
                        SimpleMessage(
                            message=f"[Game] {eliminated} was voted out! They were a {self.agent_to_role[eliminated]}."
                        ),
                    )
                    _gen_logger.info(
                        f"{eliminated} was voted out! They were a {self.agent_to_role[eliminated]}."
                    )
                # Clear votes
                self.internal_state["votes"] = {}


# ============================================================================
# Setup helpers
# ============================================================================


def ensure_agent_profile(config: Dict[str, Any]) -> AgentProfile:
    """Create or retrieve agent profile."""
    name = config.get("name", "")
    role = config.get("role", "")

    first_name, _, last_name = name.partition(" ")
    if not last_name:
        last_name = ""

    # Try to find existing
    try:
        existing = AgentProfile.find(
            (AgentProfile.first_name == first_name)
            & (AgentProfile.last_name == last_name)
        ).all()
        if existing:
            return AgentProfile.get(existing[0].pk)
    except Exception:
        pass

    # Create new
    role_secret = config.get("role_secrets", {}).get(role, "")
    profile = AgentProfile(
        first_name=first_name,
        last_name=last_name,
        secret=role_secret,
    )
    profile.save()
    return profile


def create_environment(
    env_profile: EnvironmentProfile, model_name: str, config: Dict[str, Any]
) -> SpyfallEnv:
    """Create spyfall game environment."""
    return SpyfallEnv(
        env_profile=env_profile,
        config=config,
        model_name=model_name,
        action_order="round-robin",
        evaluators=[SpyfallGameEndEvaluator(max_turn_number=MAX_TURNS)],
        terminal_evaluators=[],
        hide_unknown=True,
    )


def create_agents(
    agent_profiles: List[AgentProfile],
    env_profile: EnvironmentProfile,
    model_name: str | Dict[str, str] | List[str],
    config: Dict[str, Any],
) -> List[LLMAgent]:
    """Create LLM agents."""
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
        # Calculate secrets
        agent_name = f"{profile.first_name}{' ' + profile.last_name if profile.last_name else ''}"
        role_goal = env_profile.agent_goals[idx]

        # Get secret based on role
        role = config.get("agents", [])[idx].get("role", "")
        # role_secrets is dictionary in config
        secrets = config.get("role_secrets", {}).get(role, "")

        # Fill template
        filled_template = (
            SOCIAL_GAME_PROMPT_TEMPLATE.replace("{description}", env_profile.scenario)
            .replace("{secret}", f"Your secret info: {secrets}")
            .replace(
                "{goal}",
                role_goal,
            )
            .replace(
                "{reflection}",
                _reflection_text
                if idx < len(config.get("agents", []))
                and config["agents"][idx].get("include_reflection")
                else "",
            )
        )

        # Determine model
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
        agent.goal = env_profile.agent_goals[idx]
        agents.append(agent)
    return agents


def _pin_location_from_pool(config: Dict[str, Any]) -> None:
    """Pick a location from the configured pool and substitute it into
    the Non-Spy role_secret template.

    Deterministic from config["seed"]. Skipped if role_secrets["Non-Spy"]
    doesn't contain the {location} placeholder (i.e., already pinned) or
    if no location_pool is configured.
    """
    pool = config.get("location_pool", [])
    if not pool:
        return
    role_secrets = config.get("role_secrets", {})
    nonspy_template = role_secrets.get("Non-Spy", "")
    if "{location}" not in nonspy_template:
        # Already pinned to a literal location; the Spy still gets the list of possible locations.
        config["role_secrets"] = dict(
            role_secrets,
            Spy=role_secrets.get("Spy", "").replace(
                "{location_list}", ", ".join(sorted(pool))
            ),
        )
        return

    rng = random.Random(config.get("seed"))
    sorted_pool = sorted(pool)
    location = rng.choice(sorted_pool)
    new_role_secrets = dict(role_secrets)
    new_role_secrets["Non-Spy"] = nonspy_template.replace("{location}", location)
    # The Spy guesses from the list of possible locations (the rulebook's location spread).
    new_role_secrets["Spy"] = role_secrets.get("Spy", "").replace(
        "{location_list}", ", ".join(sorted_pool)
    )
    config["role_secrets"] = new_role_secrets
    config["location"] = location


def _location_of(config: Dict[str, Any]) -> str:
    """This episode's location: pinned by _pin_location_from_pool, or read from a roster's literal Non-Spy secret."""
    if config.get("location"):
        return str(config["location"])
    m = re.search(
        r"The location is:\s*(.+?)\.?$",
        config.get("role_secrets", {}).get("Non-Spy", ""),
    )
    return m.group(1).strip() if m else ""


def prepare_scenario(
    env_model_name: str,
    agent_model_name: str | Dict[str, str] | List[str],
    config: Dict[str, Any] | None = None,
) -> tuple[SocialDeductionGame, List[LLMAgent]]:
    """Load config and create profiles."""
    if config is None:
        config = load_config(CONFIG_PATH)
    _pin_location_from_pool(config)

    # Create agent profiles
    agent_profiles = []
    agent_goals = []
    for entry in config.get("agents", []):
        profile = ensure_agent_profile(entry)
        agent_profiles.append(profile)

        role_goal = config.get("role_goals", {}).get(entry.get("role", ""), "")
        agent_goals.append(role_goal)

    # Create environment profile
    agent_names = [entry.get("name", "") for entry in config.get("agents", [])]
    scenario = config.get("description", "Spyfall game").format(
        agent_names=", ".join(agent_names)
    )
    env_profile = EnvironmentProfile(
        scenario=scenario,
        relationship=RelationshipType.acquaintance,
        agent_goals=agent_goals,
        tag="spyfall",
    )
    env_profile.save()

    env = create_environment(env_profile, env_model_name, config)
    agents = create_agents(agent_profiles, env_profile, agent_model_name, config)
    return env, agents


def print_roster(config: Dict[str, Any]) -> None:
    """Print game roster."""
    print("Participants & roles:")
    for entry in config.get("agents", []):
        name = entry.get("name", "Unknown")
        role = entry.get("role", "Unknown")
        print(f" - {name}: {role}")


# ============================================================================
# Main
# ============================================================================


def get_model_names(config: Dict[str, Any]) -> Dict[str, str]:
    """Extract model names from config. Enforces strict requirement."""
    model_map = {}
    for entry in config.get("agents", []):
        name = entry.get("name")
        model = entry.get("agent_model")
        if not name:
            continue
        if not model:
            raise ValueError(
                f"Agent '{name}' missing 'agent_model' in config configuration."
            )
        model_map[name] = model
    return model_map


async def main() -> None:
    """Run spyfall game."""
    import argparse

    parser = argparse.ArgumentParser(description="Run Spyfall game.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(CONFIG_PATH),
        help="Path to configuration file",
    )
    parser.add_argument(
        "--roster",
        type=str,
        default=str(BASE_DIR / "roster.json"),
        help="Path to roster file",
    )
    args = parser.parse_args()

    config_path = args.config
    roster_path = args.roster

    config = load_config(config_path)
    roster = load_config(roster_path)

    # Merge roster into config
    config["agents"] = roster.get("agents", [])

    env_model_name = "gpt-4o"
    agent_model_name = get_model_names(config)

    # Setup
    env, agents = prepare_scenario(env_model_name, agent_model_name, config)

    # Display roster
    # Config already loaded above
    print("🕵️ Spyfall")
    print("=" * 60)
    print_roster(config)
    print("=" * 60)

    # Run game
    await arun_one_episode(
        env=env,
        agent_list=agents,
        omniscient=False,
        script_like=False,
        json_in_script=False,
        tag="test_spyfall",
        push_to_db=True,
    )


if __name__ == "__main__":
    # Configure logging for standalone execution
    LOG_FILE = BASE_DIR / "spyfall_game_debug.log"
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
