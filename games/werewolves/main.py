"""Launcher for the Duskmire Werewolves social game scenario."""

from __future__ import annotations

import asyncio
import copy
import os
import re
from pathlib import Path
import logging
from typing import Any, Dict, List, Tuple
from collections import Counter

from rich.logging import RichHandler
import redis
from redis_om import get_redis_connection

from sotopia.agents import LLMAgent
from sotopia.agents.llm_agent import Agents
from sotopia.database.persistent_profile import (
    AgentProfile,
    EnvironmentProfile,
    RelationshipType,
)
from sotopia.envs import SocialDeductionGame
from sotopia.envs.social_game import (
    ActionHandler,
    load_config,
    SOCIAL_GAME_PROMPT_TEMPLATE,
)
from sotopia.envs.evaluators import SocialGameEndEvaluator
from sotopia.server import arun_one_episode
from sotopia.messages import AgentAction, SimpleMessage, Message, Observation

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"

os.environ.setdefault("REDIS_OM_URL", "redis://:@localhost:6379")
redis.Redis(host="localhost", port=6379)

# Loggers (configured in main if running standalone)
_gen_logger = logging.getLogger("sotopia.generation")
_env_logger = logging.getLogger("sotopia.envs.social_game")


# ============================================================================
# Werewolf game-end evaluator
# ============================================================================


class WerewolfGameEndEvaluator(SocialGameEndEvaluator):
    """Evaluator that checks werewolf win conditions."""

    def _check_win_conditions(  # type: ignore[override]
        self, env: Any, turn_number: int, messages: List[tuple[str, Message]]
    ) -> tuple[bool, str, Dict[str, float]]:
        """Check if game has ended based on werewolf win conditions."""
        # Count alive players by team
        team_counts: Dict[str, int] = {}
        for agent_name, alive in env.agent_alive.items():
            if alive:
                role = env.agent_to_role.get(agent_name, "")
                team = env.role_to_team.get(role, "")
                team_counts[team] = team_counts.get(team, 0) + 1

        # Check end conditions from config
        end_conditions = env._config.get("end_conditions", [])
        for condition in end_conditions:
            cond_type = condition.get("type")

            if cond_type == "team_eliminated":
                team = condition.get("team", "")
                if team_counts.get(team, 0) == 0:
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
                team1 = condition.get("team", "")
                team2 = condition.get("other", "")
                if team_counts.get(team1, 0) >= team_counts.get(team2, 0):
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

        # Win conditions first: a vote or night that decides the game on the capped turn is a win, not a draw.
        terminated, reason, rewards = self._check_win_conditions(
            env, turn_number, messages
        )
        if not terminated and turn_number >= self.max_turn_number:
            return [("environment", (("terminated", True), "Max turns reached"))]

        response: List[tuple[str, tuple[tuple[str, int | float | bool], str]]] = [
            ("environment", (("terminated", terminated), reason))
        ]

        if terminated and rewards:
            agent_names = list(env.agents)  # Ensure order matches env.agents
            for agent_name, reward in rewards.items():
                # Find index of agent in env.agents
                try:
                    idx = agent_names.index(agent_name)
                    generic_key = f"agent_{idx + 1}"  # e.g. agent_1, agent_2
                    response.append((generic_key, (("complete_rating", reward), "")))
                except ValueError:
                    # Should not happen if agent_name is in env.agents
                    continue

        return response


# ============================================================================
# Werewolf-specific game logic
# ============================================================================


class WerewolfActionHandler(ActionHandler):
    """Handles actions for the Werewolf game."""

    def handle_action(
        self, env: SocialDeductionGame, agent_name: str, action: AgentAction
    ) -> None:
        """Handle a single action from an agent based on current state."""

        # Only a living player can be voted for, killed, inspected, saved or poisoned. A move naming anyone else is
        # an abstention (day vote, announced to all, since votes are public) or has no effect (night action, a
        # private notice to the acting role; no potion is used up), so the mistake stays visible. A failed model
        # call arrives as 'none' and is read as an empty reply; every phase says what that means.
        if not _acting(env, agent_name):
            return
        text = action.argument if action.action_type == "action" else ""
        names = list(env.agents)

        if env.current_state == "Day_vote":
            # Collect votes for elimination
            if "votes" not in env.internal_state:
                env.internal_state["votes"] = {}

            target, mentioned = read_named_player(text, names)
            if target and env.agent_alive.get(target, False):
                env.internal_state["votes"][agent_name] = target
                return
            if target:
                reason = f"{agent_name}'s vote for {target} does not count: {target} is already out of the game."
            elif mentioned:
                reason = f"{agent_name}'s vote names several players ({', '.join(mentioned)}) and no single 'vote NAME'."
            elif _has_verb(text, _VOTE_VERB):
                reason = f"{agent_name}'s vote names no player."
            else:
                reason = f"{agent_name} cast no vote."
            env.recv_message(
                "Environment",
                SimpleMessage(message=f"[Game] {reason} Counted as an abstention."),
            )

        elif env.current_state == "Night_werewolf":
            # Werewolves choose kill target. Any living player may be chosen, a Werewolf included (naming a
            # partner or oneself is allowed play, e.g. to draw out the Witch's save).
            role = env.agent_to_role.get(agent_name, "")
            if role == "Werewolf":
                wolves = [
                    a for a in env.agents if env.agent_to_role.get(a) == "Werewolf"
                ]
                target, mentioned = read_named_player(text, names, _KILL_VERB)
                problem = ""
                if target and not env.agent_alive.get(target, False):
                    problem = f"{agent_name}'s kill target {target} is already dead."
                elif not target:
                    problem = _night_problem(
                        agent_name, text, mentioned, "kill", _KILL_VERB
                    )
                if problem:
                    env.recv_message(
                        "Environment",
                        SimpleMessage(
                            message=f"[Private to Werewolves] {problem} This proposal does not count."
                        ),
                        receivers=wolves,
                    )
                else:
                    if "kill_target_proposals" not in env.internal_state:
                        env.internal_state["kill_target_proposals"] = {}
                    env.internal_state["kill_target_proposals"][agent_name] = target
                    # Update the werewolf kill result
                    vote_counts = Counter(
                        env.internal_state["kill_target_proposals"].values()
                    )
                    max_votes = max(vote_counts.values())
                    # Get all targets with max votes
                    candidates = [t for t, c in vote_counts.items() if c == max_votes]
                    # Break tie by the roster-seeded draw (kept
                    # secret: no announcement at night).
                    env.internal_state["kill_target"] = (
                        candidates[0]
                        if len(candidates) == 1
                        else env.break_tie(candidates)
                    )

        elif env.current_state == "Night_seer":
            # Seer inspects someone
            role = env.agent_to_role.get(agent_name, "")
            if role == "Seer":
                target, mentioned = read_named_player(text, names, _INSPECT_VERB)
                if target and env.agent_alive.get(target, False):
                    # Reveal target's role to seer
                    target_role = env.agent_to_role.get(target, "Unknown")
                    target_team = env.role_to_team.get(target_role, "Unknown")
                    note = f"{target} is on team: {target_team}"
                elif target:
                    note = f"{target} is already dead; you can only inspect a living player. No inspection tonight."
                else:
                    problem = _night_problem(
                        "You", text, mentioned, "inspect", _INSPECT_VERB
                    )
                    note = f"{problem} No inspection tonight."
                env.recv_message(
                    "Environment",
                    SimpleMessage(message=f"[Private to {agent_name}] {note}"),
                    receivers=[agent_name],
                )

        elif env.current_state == "Night_witch":
            # Witch uses potions (each available at most once across the whole game, at most one per night: the
            # common Chinese ruleset, the owner's choice; Miller's Hollow allows both on the same night).
            role = env.agent_to_role.get(agent_name, "")
            if role != "Witch":
                return
            # A negated potion command does not count ("I won't poison anyone, save Erin" is a save).
            wants = [
                p
                for p, verb in (("save", _SAVE_VERB), ("poison", _POISON_VERB))
                if _has_command(text, verb, names)
            ]
            notes: List[str] = []
            if not wants:
                if not text.strip():
                    notes.append("You sent no move. No potion used tonight.")
                elif not re.search(
                    r"\b(?:skip|pass|no\s+potions?|none|nothing)\b", text, re.IGNORECASE
                ):
                    notes.append(
                        "Your reply has no potion command ('save NAME', 'poison NAME' or 'skip'). No potion used tonight."
                    )
                # a 'skip' is a legal pass and needs no notice
            for potion in [
                p for p in wants if not env.internal_state.get(f"witch_have_{p}", False)
            ]:
                # A potion command for a potion she no longer has: nothing happens, and she is told so.
                notes.append(
                    f"You have no {potion} potion left. No potion used tonight."
                )
            usable = [
                p for p in wants if env.internal_state.get(f"witch_have_{p}", False)
            ]
            if len(usable) > 1:
                # One potion per night: the save is tried first, as before; the poison command is not used.
                notes.append(
                    "Only one potion can be used per night; your poison command was not used."
                )
            if usable:
                potion = usable[0]
                target, mentioned = read_named_player(
                    text, names, _SAVE_VERB if potion == "save" else _POISON_VERB
                )
                victim = env.internal_state.get("kill_target")
                if (
                    target
                    and env.agent_alive.get(target, False)
                    and (potion == "poison" or target == victim)
                ):
                    env.internal_state[f"witch_have_{potion}"] = False
                    env.internal_state[
                        "saved_target" if potion == "save" else "poison_target"
                    ] = target
                else:
                    if not target:
                        note = (
                            f"Your {potion} names several players ({', '.join(mentioned)})."
                            if mentioned
                            else f"Your {potion} names no player."
                        )
                    elif not env.agent_alive.get(target, False):
                        note = f"{target} is already dead; potions only work on living players."
                    elif victim:
                        # The healing potion only saves the werewolves' victim of this night.
                        note = (
                            f"The save potion only works on tonight's victim, {victim}."
                        )
                    else:
                        note = "The save potion only works on tonight's victim, and nobody was attacked tonight."
                    notes.append(f"{note} No potion used tonight.")
            for note in notes:
                env.recv_message(
                    "Environment",
                    SimpleMessage(message=f"[Private to {agent_name}] {note}"),
                    receivers=[agent_name],
                )

    def get_action_instruction(self, env: SocialDeductionGame, agent_name: str) -> str:
        """Get specific action instructions for an agent based on current state."""
        role = env.agent_to_role.get(agent_name, "")

        living = f"Players still in the game: {_living_list(env)}."

        if env.current_state == "Day_vote":
            return (
                "It is voting time. You MUST use the command 'vote NAME' to vote for a player to eliminate. "
                f"e.g. 'vote Alice'. {living} A vote for a player who is out, or a vote that names no single player, "
                "counts as an abstention. The player with the most counted votes is executed; if the top vote is "
                "tied, one of the tied players is drawn at random and executed (announced to everyone)."
            )

        elif env.current_state == "Day_discussion":
            if env._config.get("variant") == PRIVATE_REASONING:
                return PRIVATE_REASONING_INSTRUCTION

        elif env.current_state == "Night_werewolf":
            if role == "Werewolf":
                return (
                    "It is Night. You are a Werewolf. You MUST use the command 'kill NAME' to choose a target. "
                    "e.g. 'kill Bob'. You may choose any living player. If the Werewolves name different players, "
                    f"one of them is drawn at random as tonight's victim. {living}"
                )
            else:
                return "It is Night. You are sleeping."

        elif env.current_state == "Night_seer":
            if role == "Seer":
                return (
                    "It is Night. You are the Seer. You MUST use the command 'inspect NAME' to check a player's "
                    f"team. e.g. 'inspect Charlie'. {living}"
                )
            else:
                return "It is Night. You are sleeping."

        elif env.current_state == "Night_witch":
            if role == "Witch":
                if env.internal_state.get(
                    "witch_have_poison", True
                ) and env.internal_state.get("witch_have_save", True):
                    use_potion_guide = "You can use 'save NAME' or 'poison NAME', at most one potion per night. If you don't want to use potions, you can put 'skip' in the argument of action."
                elif env.internal_state.get("witch_have_poison", True):
                    use_potion_guide = "You can use 'poison NAME'. If you don't want to use potions, you can put 'skip' in the argument of action."
                elif env.internal_state.get("witch_have_save", True):
                    use_potion_guide = "You can use 'save NAME'. If you don't want to use potions, you can put 'skip' in the argument of action."
                else:
                    use_potion_guide = (
                        "You can't use any potions as you don't have any left."
                    )
                killed_message = ""
                # kill_target only ever names a living player (proposals for the dead do not count)
                if kill_target := env.internal_state.get("kill_target", None):
                    killed_message = f"{kill_target} is killed by werewolves."
                save_rule = (
                    "The save potion only works on tonight's victim."
                    if env.internal_state.get("witch_have_save", True)
                    else ""
                )
                parts = [
                    "It is Night. You are the Witch.",
                    use_potion_guide,
                    save_rule,
                    killed_message,
                    living,
                ]
                return " ".join(p for p in parts if p)
            else:
                return "It is Night. You are sleeping."

        return ""


def _acting(env: SocialDeductionGame, agent_name: str) -> bool:
    """Whether the engine asked this player to act in the step being processed."""
    try:
        return bool(env.action_mask[env.agents.index(agent_name)])
    except (ValueError, IndexError):
        return False


def _living_list(env: SocialDeductionGame) -> str:
    return ", ".join(a for a in env.agents if env.agent_alive.get(a, False))


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


_KILL_VERB = r"kill(?:s|ing)?"
_INSPECT_VERB = r"inspect(?:s|ing)?|check(?:s|ing)?"
_SAVE_VERB = r"sav(?:e|es|ing)"
_POISON_VERB = r"poison(?:s|ing)?"


def _has_verb(text: str, verb: str) -> bool:
    return bool(re.search(rf"\b(?:{verb})\b", text or "", re.IGNORECASE))


def _has_command(text: str, verb: str, names: List[str]) -> bool:
    """Whether the reply gives the command at least once without negating it ("don't poison" does not count)."""
    text = (text or "").translate(_QUOTES)
    return any(
        not _negated_command(text, m.start(), verb, names)
        for m in re.finditer(rf"\b(?:{verb})\b", text, re.IGNORECASE)
    )


def _night_problem(
    who: str, text: str, mentioned: List[str], command: str, verb: str
) -> str:
    """Why a night action names no single player, for the private notice."""
    subject = "You" if who == "You" else who
    if not (text or "").strip():
        return f"{subject} sent no move."
    if mentioned:
        return f"{subject} named several players ({', '.join(mentioned)}) and no single '{command} NAME'."
    if _has_verb(text, verb):
        return f"{subject} named no player to {command}."
    return f"{subject} did not use '{command} NAME' and named no player."


class WerewolfEnv(SocialDeductionGame):
    """Werewolf game with voting, kills, and special roles."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(action_handler=WerewolfActionHandler(), **kwargs)

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
        # Witch has potions
        self.internal_state["witch_have_poison"] = True
        self.internal_state["witch_have_save"] = True
        # Werewolves have kill targets
        self.internal_state["kill_target_proposals"] = {}
        return obs

    # The werewolves' night lasts one step per werewolf dealt, in seat order, and a dead werewolf's step is idle,
    # as a dead Seer's or Witch's phase is. Otherwise the night is one step shorter once a werewolf has died, and
    # the turn numbers tell the village so although roles stay hidden on death (the common Chinese ruleset, the
    # owner's choice; Miller's Hollow turns a dead player's card face up).

    def _wolf_seats(self) -> List[str]:
        return [a for a in self.agents if self.agent_to_role.get(a) == "Werewolf"]

    def _sequential_wolf_night(self) -> bool:
        props = self._config.get("state_properties", {}).get("Night_werewolf", {})
        return bool(
            self.current_state == "Night_werewolf"
            and props.get("action_order", self.action_order) == "round-robin"
        )

    def _update_action_mask(self) -> None:
        if not self._sequential_wolf_night():
            super()._update_action_mask()
            return
        self.action_mask = [False for _ in self.agents]
        slot = getattr(self, "_state_turn_count", {}).get(self.current_state, 0)
        wolves = self._wolf_seats()
        if slot < len(wolves) and self.agent_alive.get(wolves[slot], False):
            self.action_mask[self.agents.index(wolves[slot])] = True

    def _should_transition_state(self) -> bool:
        if not self._sequential_wolf_night():
            return super()._should_transition_state()
        steps: int = getattr(self, "_state_turn_count", {}).get(self.current_state, 0)
        return steps >= len(self._wolf_seats())

    def _check_eliminations(self) -> None:
        """Apply eliminations based on collected actions."""
        # Only apply eliminations if we are about to transition state
        if not self._should_transition_state():
            return

        if self.current_state == "Day_vote":
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
                        SimpleMessage(message=f"[Game] {eliminated} was voted out!"),
                    )
                # Clear votes
                self.internal_state["votes"] = {}
                # log elimination
                _gen_logger.info(
                    f"{eliminated} was voted out! They were a {self.agent_to_role[eliminated]}."
                )
                _gen_logger.info(f"Remaining players: {self.agent_alive}")

        elif self.current_state == "Night_witch":
            # Resolve Night actions (Werewolf kill + Witch save/poison)

            kill_target = self.internal_state.get("kill_target")
            saved_target = self.internal_state.get("saved_target")
            poison_target = self.internal_state.get("poison_target")

            # Check kill
            if kill_target and self.agent_alive.get(kill_target, False):
                if kill_target != saved_target:
                    self.agent_alive[kill_target] = False
                    self.recv_message(
                        "Environment",
                        SimpleMessage(
                            message=f"[Game] {kill_target} was killed by werewolves!"
                        ),
                    )
                    _gen_logger.info(f"{kill_target} was killed by werewolves!")
                    _gen_logger.info(f"Remaining players: {self.agent_alive}")
                else:
                    self.recv_message(
                        "Environment",
                        SimpleMessage(message="[Game] An attack was prevented!"),
                    )
                    _gen_logger.info(f"An attack to {kill_target} was prevented!")
                    _gen_logger.info(f"Remaining players: {self.agent_alive}")

            # 2. Witch Poison
            if poison_target and self.agent_alive.get(poison_target, False):
                self.agent_alive[poison_target] = False
                self.recv_message(
                    "Environment",
                    SimpleMessage(
                        message=f"[Game] {poison_target} died by witch's poison!"
                    ),
                )
                _gen_logger.info(f"{poison_target} died by witch's poison!")
                _gen_logger.info(f"Remaining players: {self.agent_alive}")

            # Clear night states
            self.internal_state.pop("kill_target_proposals", None)
            self.internal_state.pop("kill_target", None)
            self.internal_state.pop("saved_target", None)
            self.internal_state.pop("poison_target", None)


# ============================================================================
# Communication-control variants (results/sec4_comm_control/README.md)
# ============================================================================

# SOTOPIA_GAME_VARIANT, read when the environment is created, names a variant; unset or empty, the game is the
# published one. Roles, night actions, votes, win rules, tie draws, the turn cap and the engine's announcements stay as
# they are in both variants.
# - no_discussion removes Day_discussion: each dawn announcement is followed by the vote. The rules text loses its
#   discussion sentences and says in one sentence that there is no discussion phase.
# - private_reasoning keeps Day_discussion and its turn order but makes it a private state (the engine's
#   visibility "private", as the Seer's and the Witch's nights): each speech reaches its author's history only. The
#   rules text loses the same sentences and says in one sentence that each player writes private reasoning nobody
#   else sees, then votes; the discussion turn's instruction says the same.
GAME_VARIANT_ENV = "SOTOPIA_GAME_VARIANT"
NO_DISCUSSION = "no_discussion"
PRIVATE_REASONING = "private_reasoning"
VARIANTS = (NO_DISCUSSION, PRIVATE_REASONING)
# (published rules text, the variant's rules text); each published text occurs once in config.json's description.
NO_DISCUSSION_RULES = [
    (
        " (2) Discussion: All living players go *one* round of public discussion to discuss and debate who might be"
        " a Werewolf. Use 'speak' to share your thoughts and suspicions. (3) Voting:",
        " (2) Voting:",
    ),
    (" (4) Twilight:", " (3) Twilight:"),
    (
        "COMMUNICATION RULES: All day discussions are public.",
        "COMMUNICATION RULES: There is no discussion phase: each day goes straight from the dawn announcement to the"
        " vote.",
    ),
    (" Day discussion is public.", ""),
    (", and during the day, they speak in the same order as they are listed.", "."),
]
PRIVATE_REASONING_SENTENCE = (
    "During the day, each player writes private reasoning that nobody else sees, then votes."
)
PRIVATE_REASONING_RULES = [
    (published, f"COMMUNICATION RULES: {PRIVATE_REASONING_SENTENCE}")
    if published.startswith("COMMUNICATION RULES:")
    else (published, replacement)
    for published, replacement in NO_DISCUSSION_RULES
]
VARIANT_RULES = {
    NO_DISCUSSION: NO_DISCUSSION_RULES,
    PRIVATE_REASONING: PRIVATE_REASONING_RULES,
}
# The Day_discussion instruction of the private_reasoning variant (the published game gives that turn none).
PRIVATE_REASONING_INSTRUCTION = "Write your reasoning; no other player sees it."


def game_variant() -> str:
    """The variant SOTOPIA_GAME_VARIANT names: '' (the published game), 'no_discussion' or 'private_reasoning'."""
    variant = os.environ.get(GAME_VARIANT_ENV, "").strip()
    if variant not in ("",) + VARIANTS:
        raise ValueError(
            f"{GAME_VARIANT_ENV}={variant!r}: Werewolves has no such variant (only "
            + " or ".join(repr(v) for v in VARIANTS)
            + ")"
        )
    return variant


def apply_variant(config: Dict[str, Any]) -> Dict[str, Any]:
    """`config` itself when no variant is set (or `config` is already the variant's). Otherwise a copy with the
    variant's rules text, marked config["variant"]: without Day_discussion (Night_witch leads to Day_vote) for
    no_discussion, with a private Day_discussion for private_reasoning."""
    variant = game_variant()
    if not variant or config.get("variant") == variant:
        return config
    config = copy.deepcopy(config)
    text = config["description"]
    for published, replacement in VARIANT_RULES[variant]:
        if text.count(published) != 1:
            raise ValueError(
                f"{variant}: the Werewolves rules text no longer has {published!r} exactly once"
            )
        text = text.replace(published, replacement)
    config["description"] = text
    if variant == NO_DISCUSSION:
        _skip_state(config, "Day_discussion")
    else:
        # The engine's private visibility: a player's message in this state reaches only the player's own history
        # (SocialDeductionGame.recv_message); the engine's own lines stay public.
        config["state_properties"]["Day_discussion"]["visibility"] = "private"
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
) -> WerewolfEnv:
    """Create werewolf game environment (the variant SOTOPIA_GAME_VARIANT names, if any)."""
    config = apply_variant(config)
    return WerewolfEnv(
        env_profile=env_profile,
        config=config,
        model_name=model_name,
        action_order="round-robin",
        # Both variants keep this cap. The no-discussion day is one turn instead of one per living player; the
        # private-reasoning day keeps the published game's turns.
        evaluators=[WerewolfGameEndEvaluator(max_turn_number=40)],
        terminal_evaluators=[],
        hide_unknown=True,
    )


def create_agents(
    agent_profiles: List[AgentProfile],
    env_profile: EnvironmentProfile,
    model_name: str | Dict[str, str] | List[str],
    config: Dict[str, Any],  # Added config here
) -> List[LLMAgent]:
    # Identify werewolves for partner info
    werewolf_goal_str = config.get("role_goals", {}).get("Werewolf", "")
    werewolves = [
        p.first_name + (" " + p.last_name if p.last_name else "")
        for p in agent_profiles
        if env_profile.agent_goals[agent_profiles.index(p)] == werewolf_goal_str
    ]

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
        secrets = ""

        # Check if agent is a werewolf (compare against the actual goal text, since
        # agent_goals[idx] stores the full goal string -- not the role label).
        is_werewolf = agent_name in werewolves

        if is_werewolf:
            partners = [w for w in werewolves if w != agent_name]
            if partners:
                secrets = f"Your secret: You are a werewolf. Your partner(s) are: {', '.join(partners)}."
            else:
                secrets = "Your secret: You are a werewolf. You have no partners."
        filled_template = (
            SOCIAL_GAME_PROMPT_TEMPLATE.replace("{description}", env_profile.scenario)
            .replace("{secret}", secrets)
            .replace(
                "{goal}",
                role_goal,  # Also replace the goal here
            )
            .replace(
                "{reflection}",
                _reflection_text
                if idx < len(config.get("agents", []))
                and config["agents"][idx].get("include_reflection")
                else "",
            )
        )
        # Determine model for this agent
        if isinstance(model_name, dict):
            # model_dict mapping "agentX" keys or names?
            # Tournament runner will likely pass a list or dict keyed by index/name.
            # Let's support a list or dict lookup.
            # Assuming 'model_name' might be a dict like {"Aurora": "gpt-4", ...}
            this_agent_model = model_name.get(
                agent_name, model_name.get("default", "gpt-4")
            )
        elif isinstance(model_name, list):
            this_agent_model = model_name[idx]
        else:
            this_agent_model = model_name

        agent = LLMAgent(
            agent_name=f"{profile.first_name}{' ' + profile.last_name if profile.last_name else ''}",
            agent_profile=profile,
            model_name=this_agent_model,
            strict_action_constraint=True,
            custom_template=filled_template,
        )
        agent.goal = env_profile.agent_goals[idx]
        agents.append(agent)
    return agents


def prepare_scenario(
    env_model_name: str,
    agent_model_name: str | Dict[str, str] | List[str],
    config: Dict[str, Any] | None = None,
) -> tuple[SocialDeductionGame, List[LLMAgent]]:
    """Load config and create profiles."""
    if config is None:
        config = load_config(CONFIG_PATH)
    config = apply_variant(config)

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
    scenario = config.get("description", "Werewolves game").format(
        agent_names=", ".join(agent_names)
    )
    env_profile = EnvironmentProfile(
        scenario=scenario,
        relationship=RelationshipType.acquaintance,
        agent_goals=agent_goals,
        tag="werewolves",
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
    """Run werewolf game."""
    import argparse

    parser = argparse.ArgumentParser(description="Run Werewolf game.")
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
    print("🌕 Duskmire Werewolves")
    print("=" * 60)
    print_roster(config)
    print("=" * 60)

    # Run game
    variant = getattr(env, "game_variant", "")
    await arun_one_episode(
        env=env,
        agent_list=agents,
        omniscient=False,
        script_like=False,
        json_in_script=False,
        tag="test_werewolves",
        push_to_db=True,
        metadata={"variant": variant} if variant else None,
    )


if __name__ == "__main__":
    # Configure logging for standalone execution
    LOG_FILE = BASE_DIR / "werewolves_game_debug.log"
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
