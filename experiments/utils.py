import json
import os
import importlib.util
import re
import sys
from typing import Any, cast


def get_game_module(game_name: str) -> Any:
    """
    Dynamically load the main module for a given game.
    Assumes standard path: examples/experimental/games/{game_name}/main.py
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    module_path = os.path.join(
        project_root, f"examples/experimental/games/{game_name}/main.py"
    )

    if not os.path.exists(module_path):
        raise ValueError(f"Game module not found at {module_path}")

    spec = importlib.util.spec_from_file_location(f"{game_name}_main", module_path)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"{game_name}_main"] = module
        spec.loader.exec_module(module)
        return module
    else:
        raise ImportError(f"Could not load module from {module_path}")


def load_roster_template(game_name: str) -> dict[str, Any]:
    """Load the base roster.json for a game."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    roster_path = os.path.join(
        project_root, f"examples/experimental/games/{game_name}/roster.json"
    )

    if not os.path.exists(roster_path):
        raise FileNotFoundError(f"Roster file not found at {roster_path}")

    with open(roster_path, "r") as f:
        return cast(dict[str, Any], json.load(f))


def load_game_config(game_name: str) -> dict[str, Any]:
    """Load the base config.json for a game."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    config_path = os.path.join(
        project_root, f"examples/experimental/games/{game_name}/config.json"
    )

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return cast(dict[str, Any], json.load(f))
    return {}


def generate_roster(
    game_name: str,
    model_a: str,
    model_b: str,
    output_dir: str,
    pair_id: str,
    swap: bool = False,
) -> str:
    """
    Generate a roster.json for a specific matchup.

    Args:
        game_name: Name of the game (e.g., 'werewolves')
        model_a: Name of Model A
        model_b: Name of Model B
        output_dir: Directory to save the generated roster
        pair_id: Identifier for the pair (e.g., 'match0')
        swap: If True, swap the assignment (Model B takes Team 1/Slot 1, Model A takes Team 2/Slot 2)

    Returns:
        Path to the generated roster file.
    """
    # Load base config (rules, states, etc.)
    base_config = load_game_config(game_name)

    # Load roster template (agents list)
    base_roster = load_roster_template(game_name)
    agents = base_roster.get("agents", [])

    if not agents:
        raise ValueError("Roster must contain 'agents' list")

    # Determine assignment logic
    # Check if teams are present
    teams = {a.get("team") for a in agents if a.get("team")}
    # Filter out None if present
    unique_teams = sorted([t for t in teams if t])

    m1 = model_b if swap else model_a
    m2 = model_a if swap else model_b

    if len(unique_teams) == 2:
        # Asymmetric / Team-based game (Werewolf, Spyfall)
        team_1 = unique_teams[0]  # e.g. "Civilians" or "Non-Spy" (alphabetical)
        team_2 = unique_teams[1]  # e.g. "Undercover" or "Spy"

        for agent in agents:
            team = agent.get("team")
            if team == team_1:
                agent["agent_model"] = m1
            elif team == team_2:
                agent["agent_model"] = m2
            else:
                # Fallback or neutral role?
                agent["agent_model"] = m1

    else:
        # Symmetric / No-team game (PD, RPS)
        for i, agent in enumerate(agents):
            if i % 2 == 0:
                agent["agent_model"] = m1
            else:
                agent["agent_model"] = m2

    # Merge: Update agents in base_config
    base_config["agents"] = agents

    # If base_config was empty (no config.json), fall back to just roster structure
    final_output = base_config if base_config else base_roster

    # Save
    os.makedirs(output_dir, exist_ok=True)
    # Sanitize model names for filename
    sanitized_m1 = m1.replace("/", "_").replace("@", "_").split("v1")[0][-10:]
    sanitized_m2 = m2.replace("/", "_").replace("@", "_").split("v1")[0][-10:]

    filename = f"roster_{game_name}_{pair_id}_{'swapped' if swap else 'normal'}_{sanitized_m1}_vs_{sanitized_m2}.json"
    output_path = os.path.join(output_dir, filename)

    with open(output_path, "w") as f:
        json.dump(final_output, f, indent=4)

    return output_path


# ---------------------------------------------------------------------------
# Failed model calls in episode logs
# ---------------------------------------------------------------------------

# Exception types that mean the request itself failed (logs written before the per-turn
# `request_failed` flag existed record only `generation.action_error`).
REQUEST_ERROR_TYPES = {
    "APIConnectionError",
    "APIError",
    "APITimeoutError",
    "AuthenticationError",
    "BadGatewayError",
    "BadRequestError",
    "ConnectionError",
    "ContextWindowExceededError",
    "InternalServerError",
    "NotFoundError",
    "PermissionDeniedError",
    "RateLimitError",
    "ServiceUnavailableError",
    "Timeout",
    "TimeoutError",
}
# Request errors that fail again the same way when the roster is replayed (HTTP 400, a prompt over the context
# window), unless the message says a self-hosted model was loading; see generate.is_permanent_request_error.
PERMANENT_ERROR_TYPES = {"BadRequestError", "ContextWindowExceededError"}
_MODEL_LOADING_WORDS = ("no models loaded", "unloaded", "not loaded", "model_not_found", "crash", "channel error", "temporarily")
# finish_reason values of a generation the server aborted; the same set as
# sotopia.generation_utils.generate.SERVER_ABORT_FINISH_REASONS (tests check they agree).
SERVER_ABORT_FINISH_REASONS = frozenset({"abort", "error"})
_EMPTY_REPLY_PREFIXES = ("empty completion", "no choices in the completion", "the server aborted the generation")
_FINISH_IN_MESSAGE = re.compile(r"finish_reason=(?:None|'([^']*)')")


def _empty_reply_failed(message: str) -> bool:
    """Whether an empty-reply error (EmptyCompletionError's message) is a request failure under the rule."""
    if message.startswith(("no choices in the completion", "the server aborted the generation")):
        return True
    m = _FINISH_IN_MESSAGE.search(message)
    finish = m.group(1) if m else None
    return finish is None or finish in SERVER_ABORT_FINISH_REASONS


def turn_request_failed(turn: dict[str, Any]) -> bool:
    """True when this turn's model call failed at the request level, so its episode is not rated.

    The rule (the same as sotopia.generation_utils.generate): a request failure is an exception from the
    completion call, a response with no choices, or a generation the server aborted (finish_reason 'abort',
    vLLM's server-side abort, or 'error'; also no finish reason and no content). Anything the server completed
    is the model's move and is rated: a reply that does not parse, an empty reply that stopped normally, a
    reply cut off at the token limit, a structured-output refusal (ModelRefusalError) or a filtered reply.

    How a log is read:
    - A turn whose recorded finish_reason is 'abort' or 'error' and that has an action_error failed.
    - An empty-reply error (EmptyCompletionError, or in the oldest logs a response that starts with its message)
      is judged by the finish reason in its message, as above. Logs written before 2026-10-05 flagged an empty
      'stop' reply as a failed request; this re-reads them by the current rule.
    - Otherwise the per-turn `generation.request_failed` flag decides when it is present.
    - Older logs carry only `generation.action_error`: a request-level exception type counts as a failed
      request, while a reply that did not parse (ValidationError) or an AssertionError from an empty reply at
      the token limit is the model's own move.
    - The oldest logs have no `generation` entry: a failed call is the turn with an empty prompt list whose
      response is the exception text, and it counts when that text names a request-level litellm error
      ("litellm.InternalServerError: ...") or is an empty-reply error that the rule calls a failure.
    """
    gen = turn.get("generation") or {}
    if gen:
        err = gen.get("action_error") or {}
        kind = str(err.get("type", ""))
        message = str(err.get("message", ""))
        if err and gen.get("finish_reason") in SERVER_ABORT_FINISH_REASONS:
            return True
        if kind == "ModelRefusalError":
            return False
        if kind == "EmptyCompletionError":
            return _empty_reply_failed(message)
        if "request_failed" in gen:
            return bool(gen["request_failed"])
        if not err:
            return False
    else:
        if turn.get("prompt") != []:
            return False
        kind, message = "", str(turn.get("response") or "")
        m = re.match(r"\s*litellm\.(\w+)", message)
        if m:
            kind = m.group(1)
        elif message.lstrip().startswith(_EMPTY_REPLY_PREFIXES):
            return _empty_reply_failed(message.lstrip())
    return kind in REQUEST_ERROR_TYPES


def turn_permanent_failure(turn: dict[str, Any]) -> bool:
    """True when the turn failed at the request level in a way a replay would repeat: HTTP 400 or a prompt over
    the context window (`generation.permanent_failure`, or for older logs the error type, unless its message says
    a self-hosted model was loading). Such a turn still keeps its episode out of the rating."""
    if not turn_request_failed(turn):
        return False
    gen = turn.get("generation") or {}
    if "permanent_failure" in gen:
        return bool(gen["permanent_failure"])
    if gen:
        err = gen.get("action_error") or {}
        kind, message = str(err.get("type", "")), str(err.get("message", ""))
    else:
        message = str(turn.get("response") or "")
        m = re.match(r"\s*litellm\.(\w+)", message)
        kind = m.group(1) if m else ""
    return kind in PERMANENT_ERROR_TYPES and not any(w in message.lower() for w in _MODEL_LOADING_WORDS)


def unrecovered_request_failures(log: dict[str, Any]) -> int:
    """Number of turns in an episode log whose model call failed at the request level."""
    return sum(turn_request_failed(t) for t in log.get("turns") or [])


def episode_failure(log: dict[str, Any]) -> str:
    """'' for a log that can be rated; 'failed' when a call failed at the request level and a replay may
    succeed (the runner replays the roster); 'permanent' when every failed call would fail again unchanged
    (the runner reports it and does not replay it). Neither kind is rated."""
    failed = [t for t in log.get("turns") or [] if turn_request_failed(t)]
    if not failed:
        return ""
    return "permanent" if all(turn_permanent_failure(t) for t in failed) else "failed"


def normalize_roster_dir(roster_dir: str) -> str:
    """The resume key's directory part, relative to experiments/rosters/: 'tournament', 'tournament/',
    './tournament', 'tournament//', 'experiments/rosters/tournament' and the absolute path of that directory are
    one directory. '' stays '' (rosters directly under experiments/rosters/<game>/)."""
    roster_dir = (roster_dir or "").strip()
    if not roster_dir:
        return ""
    rosters_root = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "experiments", "rosters")
    norm = os.path.normpath(roster_dir)
    if os.path.isabs(norm) and (norm + os.sep).startswith(rosters_root + os.sep):
        norm = os.path.relpath(norm, rosters_root)
    elif norm.startswith(os.path.join("experiments", "rosters") + os.sep):
        norm = norm[len(os.path.join("experiments", "rosters") + os.sep):]
    return "" if norm in (".", os.path.join("experiments", "rosters")) else norm


def roster_id(metadata: dict[str, Any]) -> str | None:
    """The roster a log played, as (normalized roster_dir)/(roster_file), or None when the log does not say."""
    r_file = (metadata or {}).get("roster_file", "")
    if not r_file:
        return None
    r_dir = normalize_roster_dir((metadata or {}).get("roster_dir", ""))
    return f"{r_dir}/{r_file}" if r_dir else r_file


_LOG_STAMP = re.compile(r"_(\d{8}_\d{6}_\d+)\.json$")


def log_order_key(path: str) -> tuple[str, float]:
    """Sort key for 'latest first': the timestamp in the log's file name, then the file's mtime."""
    m = _LOG_STAMP.search(os.path.basename(path))
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = 0.0
    return (m.group(1) if m else "", mtime)


def rated_logs(
    paths: list[str], failed: list[str] | None = None, duplicates: list[str] | None = None
) -> list[str]:
    """The logs to rate, one per roster, in the order given.

    A log with an unrecovered request failure is left out (appended to `failed` when given), whatever else its
    roster has. Of the clean logs of one roster (experiments.utils.roster_id) only the latest is kept, by the
    file-name timestamp and then the mtime; the others are appended to `duplicates`. Logs that name no roster are
    all kept, and so are files that do not load (the caller decides what to do with them)."""
    best: dict[str, str] = {}
    chosen: set[str] = set()
    for p in paths:
        try:
            with open(p) as f:
                data = json.load(f)
        except Exception:
            chosen.add(p)
            continue
        if unrecovered_request_failures(data):
            if failed is not None:
                failed.append(p)
            continue
        rid = roster_id(data.get("metadata") or {})
        if rid is None:
            chosen.add(p)
        elif rid not in best:
            best[rid] = p
        else:
            older, newer = sorted((best[rid], p), key=log_order_key)
            best[rid] = newer
            if duplicates is not None:
                duplicates.append(older)
    chosen |= set(best.values())
    return [p for p in paths if p in chosen]
