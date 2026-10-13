"""Launcher for the Bargaining Game (iterated ultimatum)."""

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


class BargainingEvaluator(SocialGameEndEvaluator):
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


TOKENS = 10

# Forms that name what the proposer keeps ("offer 6", "I keep 6", "6 for me") and what the responder gets
# ("you get 4", "offer you 4", "4 for you"). 'offer N' always counts the tokens the proposer keeps, as the rules
# define it, so "offer 4 to you" reads both keep 4 and keep 6 and is unclear, like "I offer 4 tokens to you".
_KEEP_FORMS = re.compile(
    r"\b(?<!you )(?:offer|offering|keep|keeping|take|taking)\s*(?:to\s+keep\s*)?[:=]?\s*['\"]?(-?\d+)\b"
    r"|\b(-?\d+)\s*(?:tokens?\s+)?for\s+me\b"
)
_GIVE_FORMS = re.compile(
    r"\b(?:you\s+(?:get|receive|keep|take|would\s+get)|give\s+you|offer\s+you|offering\s+you)\s*[:=]?\s*(-?\d+)\b"
    r"|\b(-?\d+)\s*(?:tokens?\s+)?(?:for|to)\s+you\b"
)
_NEGATION = re.compile(
    r"\b(?:not|no|never|neither|nor|cannot|don'?t|won'?t|can'?t|shouldn'?t|wouldn'?t)\b|n't\b"
)
_CLAUSE_BREAK = re.compile(r"[.;:!?,\n]|\bbut\b")
# Words that make a decision. Forms that only describe one ("accepting", "rejection", "accepted") do not count.
_DECISION = re.compile(r"\b(accept|acceptable|unacceptable|reject|decline|refuse)\b")


def is_negated(text: str, start: int) -> bool:
    """True when one of the 3 words before position `start`, in the same clause, is a negation."""
    clause = _CLAUSE_BREAK.split(text[:start])[-1]
    return bool(_NEGATION.search(" ".join(clause.split()[-3:])))


def parse_proposal(text: str) -> Tuple[int | None, str]:
    """(tokens the proposer keeps, '') or (None, why the proposal is unclear).

    'offer N' is the command (N = tokens the proposer keeps). "I keep 6" and "you get 4" are read for what they
    say; a bare number counts as 'offer N'; a negated form ("I won't offer 3") does not count. Readings that
    disagree, a number outside 0..10, or a reply that answers like a responder ('accept', 'reject') is unclear.
    """
    t = text.strip().lower().replace("’", "'")
    if _DECISION.search(t):
        return None, "it says 'accept' or 'reject', which is the Responder's move"
    readings = [
        int(m.group(1) or m.group(2)) for m in _KEEP_FORMS.finditer(t) if not is_negated(t, m.start())
    ]
    readings += [
        TOKENS - int(m.group(1) or m.group(2)) for m in _GIVE_FORMS.finditer(t) if not is_negated(t, m.start())
    ]
    if not readings:
        bare = re.fullmatch(r"\W*(-?\d+)\W*", t)
        if bare:
            readings = [int(bare.group(1))]
    if not readings:
        return None, "it names no number of tokens to keep"
    if len(set(readings)) > 1:
        return None, f"it can be read as keeping {' or '.join(str(r) for r in sorted(set(readings)))}"
    keep = readings[0]
    if not 0 <= keep <= TOKENS:
        return None, f"{keep} is not between 0 and {TOKENS}"
    return keep, ""


def parse_response(text: str) -> str | None:
    """'accept' or 'reject', the decision the reply makes; None when it makes none or both.

    Every decision word counts ('accept', 'acceptable', 'reject', 'decline', 'refuse', 'unacceptable'), a
    negated one as the other ("I do not accept this" rejects). Words that only describe a decision
    ("Accepting would hand you the lead, so I reject") do not count.
    """
    t = text.lower().replace("’", "'")
    readings = set()
    for m in _DECISION.finditer(t):
        choice = "accept" if m.group(1) in ("accept", "acceptable") else "reject"
        if is_negated(t, m.start()):
            choice = "reject" if choice == "accept" else "accept"
        readings.add(choice)
    return readings.pop() if len(readings) == 1 else None


class BargainingActionHandler(ActionHandler):
    def handle_action(
        self, env: SocialDeductionGame, agent_name: str, action: AgentAction
    ) -> None:
        if not isinstance(env, BargainingEnv) or agent_name not in env.awaiting():
            return
        replied = action.action_type in ["action", "speak"]  # anything else is a failed call
        if env.current_state == "Propose":
            keep, why = parse_proposal(action.argument) if replied else (None, "the reply failed")
            env.take_proposal(keep, why)
        elif env.current_state == "Respond":
            env.take_response(parse_response(action.argument) if replied else None)

    def get_action_instruction(self, env: SocialDeductionGame, agent_name: str) -> str:
        if not isinstance(env, BargainingEnv):
            return ""

        round_num = env.internal_state.get("round", 0)
        max_rounds = env._config.get("max_rounds", 10)
        scores = env.internal_state.get("scores", {})
        proposer, responder = env.roles()
        unclear = env.internal_state.get("unclear", 0)

        if env.current_state == "Propose":
            if agent_name == proposer:
                text = (
                    f"Round {round_num + 1}/{max_rounds}. Scores: {scores}. "
                    f"You are the PROPOSER. You have 10 tokens to split. "
                    f"Choose 'offer N' where N is how many you keep (opponent gets 10-N). "
                    f"Example: 'offer 6' means you keep 6, opponent gets 4. "
                    f"'offer N' always counts the tokens you keep, so 'offer 4 to you' is unclear. "
                    f"An unclear proposal is announced and you are asked once more; if the second is also unclear, "
                    f"your offer is drawn at random from 0 to {TOKENS}. Higher total score wins. Equal totals are a draw."
                )
                if unclear:
                    text += (
                        f" Your last proposal was unclear ({env.internal_state.get('unclear_reason', '')}), so you "
                        f"are asked once more. If this proposal is also unclear, your offer is drawn at random."
                    )
                return text
            return (
                f"Round {round_num + 1}/{max_rounds}. Scores: {scores}. "
                f"{proposer} is the Proposer this round. Waiting for their offer."
            )
        elif env.current_state == "Respond":
            offer = env.internal_state.get("current_offer")
            if offer is None:
                return ""
            if env.internal_state.get("offer_drawn"):
                offer_text = f"{proposer}'s offer was drawn at random: {proposer} keeps {offer}"
            else:
                offer_text = f"{proposer} offers to keep {offer}"
            if agent_name == responder:
                text = (
                    f"Round {round_num + 1}/{max_rounds}. Scores: {scores}. "
                    f"You are the RESPONDER. {offer_text} "
                    f"(you would get {TOKENS - offer}). "
                    f"Choose 'accept' or 'reject'. If rejected, both get 0 this round. "
                    f"Your reply is read for its decision; a reply that makes no decision, or both, is unclear: it is "
                    f"announced and you are asked once more, and if the second is also unclear, accept or reject is "
                    f"drawn at random, each equally likely. Higher total score wins. Equal totals are a draw."
                )
                if unclear:
                    text += (
                        " Your last response did not give one clear 'accept' or 'reject', so you are asked once more."
                        " If this response is also unclear, your response is drawn at random."
                    )
                return text
            return (
                f"Round {round_num + 1}/{max_rounds}. Scores: {scores}. "
                f"Waiting for {responder} to respond to your offer of keeping {offer}."
            )
        return ""


# ============================================================================
# Environment
# ============================================================================


class BargainingEnv(SocialDeductionGame):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(action_handler=BargainingActionHandler(), **kwargs)

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
            "scores": {name: 0 for name in self.agents},
            "current_offer": None,
            "current_response": None,
            "unclear": 0,  # unclear replies of the player now due, in this phase of this round
            "unclear_reason": "",
            "offer_drawn": False,
            "response_drawn": False,
            "game_over": False,
            "final_scores": {},
            "end_reason": "",
        }
        # Offers and responses drawn for twice-unclear replies; seeded from the roster so replays agree.
        self._fallback_rng = random.Random(f"{self._config.get('seed')}-fallback")
        # The base reset built the first instructions before the scores existed ("Scores: {}"): rebuild them
        for name in self.agents:
            if name in obs and self.action_handler:
                obs[name].action_instruction = self.action_handler.get_action_instruction(self, name)
        return obs

    def roles(self) -> Tuple[str, str]:
        """(proposer, responder) of the current round: seat 0 proposes in odd rounds, seat 1 in even ones."""
        agents = list(self.agents)
        proposer_idx = self.internal_state.get("round", 0) % 2
        return agents[proposer_idx], agents[1 - proposer_idx]

    def awaiting(self) -> List[str]:
        """The player whose reply the current phase still needs (none once the match is over)."""
        if self.internal_state.get("game_over"):
            return []
        proposer, responder = self.roles()
        if self.current_state == "Propose" and self.internal_state.get("current_offer") is None:
            return [proposer]
        if self.current_state == "Respond" and self.internal_state.get("current_response") is None:
            return [responder]
        return []

    def _unclear(self, who: str, what: str, rule: str) -> bool:
        """Count an unclear reply and announce it; True when it is the second one (the move is then drawn)."""
        self.internal_state["unclear"] = self.internal_state.get("unclear", 0) + 1
        if self.internal_state["unclear"] >= 2:
            return True
        round_num = self.internal_state.get("round", 0) + 1
        self.recv_message(
            "Environment",
            SimpleMessage(message=f"[Round {round_num}] {who}'s {what} {rule}, so {who} is asked once more."),
        )
        return False

    def take_proposal(self, keep: int | None, why: str) -> None:
        """Hold a clear offer and announce it; ask an unclear (or failed) proposal once more, then draw the offer
        uniformly from 0..10. Only the parsed offer is announced, never the proposal's text."""
        proposer, responder = self.roles()
        round_num = self.internal_state.get("round", 0) + 1
        if keep is None:
            self.internal_state["unclear_reason"] = why
            if not self._unclear(proposer, "proposal", "was unclear"):
                return
            keep = self._fallback_rng.randint(0, TOKENS)
            self.internal_state["offer_drawn"] = True
            line = (
                f"[Round {round_num}] {proposer}'s second proposal was also unclear, so {proposer}'s offer is drawn "
                f"at random: {proposer} keeps {keep}, {responder} would get {TOKENS - keep}."
            )
        else:
            line = f"[Round {round_num}] {proposer} offers to keep {keep}; {responder} would get {TOKENS - keep}."
        self.internal_state["current_offer"] = keep
        self.internal_state["unclear"] = 0
        self.internal_state["unclear_reason"] = ""
        self.recv_message("Environment", SimpleMessage(message=line))

    def take_response(self, choice: str | None) -> None:
        """Hold a clear response; ask an unclear (or failed) one once more, then draw accept or reject, 50/50."""
        _, responder = self.roles()
        if choice is None:
            if not self._unclear(responder, "response", "did not give one clear 'accept' or 'reject'"):
                return
            choice = self._fallback_rng.choice(("accept", "reject"))
            self.internal_state["response_drawn"] = True
        self.internal_state["current_response"] = choice
        self.internal_state["unclear"] = 0

    def _update_action_mask(self) -> None:
        """Only the player whose reply is due acts."""
        super()._update_action_mask()
        if self.current_state in ("Propose", "Respond"):
            due = self.awaiting()
            self.action_mask = [name in due for name in self.agents]

    def _should_transition_state(self) -> bool:
        # Stay in the phase while its player is asked once more; nothing follows the last round.
        if self.internal_state.get("game_over"):
            return False
        if self.current_state == "Propose" and self.internal_state.get("current_offer") is None:
            return False
        if self.current_state == "Respond" and self.internal_state.get("unclear"):
            return False
        return super()._should_transition_state()

    def _check_eliminations(self) -> None:
        if self.current_state != "Respond" or self.internal_state.get("current_response") is None:
            return

        agents = list(self.agents)
        round_num = self.internal_state.get("round", 0)
        proposer, responder = self.roles()
        offer = self.internal_state["current_offer"]
        response = self.internal_state["current_response"]
        scores = self.internal_state["scores"]
        if self.internal_state.get("offer_drawn"):
            offered = f"{proposer}'s offer was drawn at random: keep {offer}"
        else:
            offered = f"{proposer} offered to keep {offer}"
        if self.internal_state.get("response_drawn"):
            decided = f"{responder}'s second response was also unclear, so it is drawn at random: {response.upper()}ED"
        else:
            decided = f"{responder} {response.upper()}ED"

        if response == "accept":
            proposer_gain = offer
            responder_gain = TOKENS - offer
            scores[proposer] += proposer_gain
            scores[responder] += responder_gain
            msg = (
                f"[Round {round_num + 1}] {offered}. {decided}. "
                f"{proposer}+{proposer_gain}, {responder}+{responder_gain}. Scores: {scores}"
            )
        else:
            msg = f"[Round {round_num + 1}] {offered}. {decided}. Both get 0. Scores: {scores}"

        self.internal_state["round"] = round_num + 1
        self.internal_state["current_offer"] = None
        self.internal_state["current_response"] = None
        self.internal_state["unclear"] = 0
        self.internal_state["unclear_reason"] = ""
        self.internal_state["offer_drawn"] = False
        self.internal_state["response_drawn"] = False

        self.recv_message("Environment", SimpleMessage(message=msg))

        max_rounds = self._config.get("max_rounds", 10)
        if self.internal_state["round"] >= max_rounds:
            s1, s2 = scores[agents[0]], scores[agents[1]]
            if s1 > s2:
                final = {agents[0]: 1.0, agents[1]: -1.0}
                reason = f"Game over. {agents[0]} wins! Final: {scores}"
            elif s2 > s1:
                final = {agents[0]: -1.0, agents[1]: 1.0}
                reason = f"Game over. {agents[1]} wins! Final: {scores}"
            else:
                final = {agents[0]: 0.0, agents[1]: 0.0}
                reason = f"Game over. Tie! Final: {scores}"

            self.internal_state["game_over"] = True
            self.internal_state["final_scores"] = final
            self.internal_state["end_reason"] = reason
            self.recv_message(
                "Environment", SimpleMessage(message=f"[Game] {reason}")
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
) -> BargainingEnv:
    return BargainingEnv(
        env_profile=env_profile,
        config=config,
        model_name=model_name,
        # Safety cap only: each round takes two steps, or up to four when replies are asked again.
        evaluators=[BargainingEvaluator(max_turn_number=4 * config.get("max_rounds", 10))],
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
    scenario = config.get("description", "Bargaining").format(
        agent_names=", ".join(agent_names)
    )
    env_profile = EnvironmentProfile(
        scenario=scenario,
        relationship=RelationshipType.acquaintance,
        agent_goals=agent_goals,
        tag="bargaining",
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

    parser = argparse.ArgumentParser(description="Run Bargaining game.")
    parser.add_argument("--config", type=str, default=str(CONFIG_PATH))
    parser.add_argument("--roster", type=str, default=str(BASE_DIR / "roster.json"))
    args = parser.parse_args()

    config = load_config(args.config)
    roster = load_config(args.roster)
    config["agents"] = roster.get("agents", [])

    agent_model_name = get_model_names(config)
    env, agents = prepare_scenario("gpt-4o", agent_model_name, config)

    print("Bargaining Game (Iterated Ultimatum)")
    print("=" * 60)
    print_roster(config)
    print("=" * 60)

    await arun_one_episode(
        env=env,
        agent_list=agents,
        omniscient=False,
        script_like=False,
        json_in_script=False,
        tag="test_bargaining",
        push_to_db=True,
    )


if __name__ == "__main__":
    LOG_FILE = BASE_DIR / "bargaining_game_debug.log"
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
