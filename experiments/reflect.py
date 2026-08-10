"""
Play-Reflect-Transfer pipeline for self-improvement experiments.

Usage:
    # Single game reflection
    python -m experiments.reflect \
        --log-dir logs/reflect/werewolf_r1 \
        --model gpt-5 \
        --output experiments/reflections/gpt5_werewolf_r1.txt

    # Multi-game reflection (pass multiple --log-dir)
    python -m experiments.reflect \
        --log-dir logs/reflect/werewolf_r1 \
        --log-dir logs/reflect/chameleon_r1 \
        --log-dir logs/reflect/resistance_r1 \
        --model gpt-5 \
        --output experiments/reflections/gpt5_multigame_r1.txt
"""

import argparse
import asyncio
import glob
import json
import os
import re

from litellm import acompletion, completion

from experiments.utils import load_game_config


def get_game_description(game_name: str) -> str:
    """Load game description from config.json."""
    config = load_game_config(game_name)
    return config.get("description")


def _extract_state_events(turns: list[dict]) -> dict[int, list[str]]:
    """Extract `[Game] ...` system events keyed by turn_number.

    Every agent's prompt embeds a running ``Conversation Starts:`` history that
    contains the public game-state announcements (phase transitions, kills,
    vote outcomes, etc.). We parse that history from the latest available
    prompt and bucket the [Game] lines per turn so they can be interleaved
    with player actions in the reflection summary.
    """
    if not turns:
        return {}
    # Find the latest turn whose prompt actually contains a Conversation Starts block.
    # Some turns may have an empty prompt list (system / phase-transition entries).
    history = ""
    for t in reversed(turns):
        prompt = t.get("prompt") or []
        if not prompt:
            continue
        content = prompt[0].get("content", "") if isinstance(prompt[0], dict) else ""
        if "Conversation Starts:" in content:
            history = content
            break
    if not history:
        return {}
    start = history.find("Conversation Starts:")
    end = history.find("Your available action")
    if start < 0 or end < 0:
        return {}
    history = history[start:end]

    events: dict[int, list[str]] = {}
    current_turn = -1
    for raw in history.split("\n"):
        line = raw.strip()
        if not line:
            continue
        # "Turn #N:" anchors what follows to that turn number
        if line.startswith("Turn #"):
            try:
                tnum = int(line.split("Turn #", 1)[1].split(":", 1)[0])
                current_turn = tnum
            except (ValueError, IndexError):
                continue
            rest = line.split(":", 1)[1].strip() if ":" in line else ""
            if rest.startswith("[Game]"):
                events.setdefault(current_turn, []).append(rest)
        elif line.startswith("[Game]") and current_turn >= 0:
            events.setdefault(current_turn, []).append(line)
    return events


def format_game_summary(log_data: dict, game_idx: int) -> str:
    """Format a single game into a trajectory summary with game-state events.

    Game rules are emitted once globally via build_rules_section(). For each
    turn we show the actor's action, and we interleave any ``[Game] ...``
    state announcements (phase changes, kills, vote results) that occurred
    at that turn.
    """
    turns = log_data.get("turns", [])
    rewards = log_data.get("rewards", [])
    mm = log_data.get("model_mapping", {})
    agents = list(mm.keys())
    game_name = log_data.get("metadata", {}).get("game_name", "Unknown")

    r_vals = [float(r[0]) if isinstance(r, list) else float(r) for r in rewards]
    winners = [agents[i] for i in range(len(agents)) if r_vals[i] > 0]
    losers = [agents[i] for i in range(len(agents)) if r_vals[i] < 0]

    state_events = _extract_state_events(turns)

    lines = [f"=== Game {game_idx + 1}: {game_name} ==="]
    lines.append(f"Players: {', '.join(agents)}")
    lines.append(f"Winners: {', '.join(winners) if winners else 'None'}")
    lines.append(f"Losers: {', '.join(losers) if losers else 'None'}")
    lines.append("")

    last_emitted_turn = -1
    for t in turns:
        tnum = t["turn_number"]
        # Emit any state events for new turn numbers we haven't seen yet
        for past_tnum in range(last_emitted_turn + 1, tnum + 1):
            for ev in state_events.get(past_tnum, []):
                lines.append(f"  Turn {past_tnum:>2} {ev}")
        last_emitted_turn = tnum

        name = t.get("agent_name", "")
        response = t.get("response", "")
        try:
            action = json.loads(response)
            action_str = action.get("argument", response)
        except (json.JSONDecodeError, TypeError):
            action_str = response[:200] if response else "(no response)"
        lines.append(f"  Turn {tnum:>2} [{name}]: {action_str}")

    # Flush any trailing state events beyond the last actor turn
    max_turn = max(state_events) if state_events else last_emitted_turn
    for past_tnum in range(last_emitted_turn + 1, max_turn + 1):
        for ev in state_events.get(past_tnum, []):
            lines.append(f"  Turn {past_tnum:>2} {ev}")

    lines.append("")
    return "\n".join(lines)


def build_rules_section(game_names: list[str]) -> str:
    """Emit one rules block per distinct game name, in first-occurrence order."""
    seen: list[str] = []
    for g in game_names:
        if g not in seen:
            seen.append(g)
    lines = ["=== Game rules ==="]
    for g in seen:
        lines.append(f"--- {g} ---")
        lines.append(get_game_description(g))
        lines.append("")
    return "\n".join(lines)


def build_reflection_prompt(
    game_summaries: str, num_games: int, game_names: list[str], prior_reflection: str = ""
) -> str:
    """Build the reflection prompt for single or multi-game reflection."""
    # Per-game counts in first-occurrence order: "10 werewolves, 10 chameleon, ..."
    counts: dict[str, int] = {}
    order: list[str] = []
    for g in game_names:
        if g not in counts:
            order.append(g)
        counts[g] = counts.get(g, 0) + 1
    games_desc = ", ".join(f"{counts[g]} {g}" for g in order)

    prior_section = ""
    if prior_reflection:
        prior_section = f"""You previously wrote the following strategic playbook for yourself:

--- YOUR PREVIOUS PLAYBOOK ---
{prior_reflection}
--- END PREVIOUS PLAYBOOK ---

You then played {num_games} more games using this playbook ({games_desc}). Each player was an independent instance of you and only saw its own role's private information.
Based on these new games, **revise your playbook**. Keep rules that worked, remove or modify rules that didn't help, and add new insights. Output the complete revised playbook (not just the changes).

"""
    else:
        prior_section = f"""You just played {num_games} self-play games ({games_desc}). Each player was an independent instance of you and only saw its own role's private information; the transcripts below reveal the hidden moves of every role, but the players themselves did not have this view during play.

"""

    return f"""{prior_section}Below are the full trajectories of all {num_games} games, showing every player's actions and the outcome:

{game_summaries}

Write an **internal monologue** of transferable social reasoning skills. Frame your insights around general capabilities that apply across many social games, such as:

- **Deception**: hiding your true intentions, bluffing, maintaining a consistent false persona
- **Detection**: identifying when others are lying, spotting inconsistencies, reading behavioral patterns
- **Persuasion**: convincing others to act in your interest, building credibility, framing arguments
- **Information management**: when to reveal, withhold, or fabricate information; timing of disclosures
- **Coalition dynamics**: building alliances, breaking enemy alliances, knowing when to lead vs follow
- **Timing and patience**: when to act early vs wait, when to commit vs stay flexible

Requirements:
- Write in first person ("I should...", "When I need to hide information...", "A pattern I noticed is...")
- Derive insights from the games above, but write the rules so they apply beyond any specific game
- Focus on actionable lessons, not abstract observations
- Do NOT reference specific game numbers (e.g., "Game 3", "Games 5-8"). Your future self will not have access to these transcripts, so such references would be meaningless

This monologue will be prepended to your system prompt in future social games. Write it so that reading it once before any social strategy game will meaningfully improve your play.
"""


def build_placebo_prompt() -> str:
    """Placebo-control prompt: same writing instructions as the real reflection prompt,
    but without any trajectory grounding. The model is asked to write the playbook
    from general knowledge, before having played any games. Used to test whether
    SPaRTan's effects depend on trajectory-derived content vs. any length-matched
    advice-list text injected into the system prompt."""
    return """Write 1200 words about an **internal monologue** of social reasoning skills. Frame your insights around general capabilities that apply across many social games, such as:

- **Deception**: hiding your true intentions, bluffing, maintaining a consistent false persona
- **Detection**: identifying when others are lying, spotting inconsistencies, reading behavioral patterns
- **Persuasion**: convincing others to act in your interest, building credibility, framing arguments
- **Information management**: when to reveal, withhold, or fabricate information; timing of disclosures
- **Coalition dynamics**: building alliances, breaking enemy alliances, knowing when to lead vs follow
- **Timing and patience**: when to act early vs wait, when to commit vs stay flexible

Requirements:
- Write in first person ("I should...", "When I need to hide information...", "A pattern I noticed is...")
- Focus on actionable lessons, not abstract observations
- Write the rules so they apply beyond any specific game

This monologue will be prepended to your system prompt in future social games. Write it so that reading it once before any social strategy game will meaningfully improve your play.
"""


async def run_reflection(
    log_dirs: list[str],
    model: str,
    output_path: str,
    log_pattern: str = "*.json",
    max_per_dir: int = 0,
    prior_reflection_path: str = "",
) -> None:
    """Read game logs from one or more directories, format trajectories, prompt model for reflection."""

    # Collect all log files from all directories
    log_files = []
    for log_dir in log_dirs:
        pattern = os.path.join(log_dir, log_pattern)
        found = sorted(glob.glob(pattern))
        if max_per_dir > 0 and len(found) > max_per_dir:
            found = found[:max_per_dir]
        print(f"  {log_dir}: {len(found)} logs")
        log_files.extend(found)

    if not log_files:
        print("No logs found.")
        return

    print(f"Total: {len(log_files)} game logs")

    # Format all game summaries
    summaries = []
    game_names = []
    for i, f in enumerate(log_files):
        data = json.load(open(f))
        game_name = data.get("metadata", {}).get("game_name", "Unknown")
        game_names.append(game_name)
        summaries.append(format_game_summary(data, i))

    # One rules block per distinct game, prepended before the trajectories
    all_summaries = build_rules_section(game_names) + "\n" + "\n".join(summaries)

    # Load prior reflection if iterating
    prior_reflection = ""
    if prior_reflection_path:
        if not os.path.exists(prior_reflection_path):
            raise FileNotFoundError(f"Prior reflection not found: {prior_reflection_path}")
        with open(prior_reflection_path) as f:
            prior_reflection = f.read().strip()
        print(f"Prior reflection loaded: {prior_reflection_path} ({len(prior_reflection)} chars)")

    prompt = build_reflection_prompt(all_summaries, len(log_files), game_names, prior_reflection)

    print(f"Games: {', '.join(sorted(set(game_names)))}")
    print(f"Prompt length: {len(prompt)} chars")
    print(f"Calling {model} for reflection...")

    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }

    if model.startswith("custom"):
        base_url = model.split("@")[1]
        api_key = os.environ.get("CUSTOM_API_KEY", "EMPTY")
        clean_model = model.split("@")[0].replace("custom/", "openai/")
        kwargs.update({"model": clean_model, "base_url": base_url, "api_key": api_key})

    response = await acompletion(**kwargs)
    raw_reflection = response.choices[0].message.content

    # Strip reasoning-model <think>...</think> blocks.
    reflection = re.sub(r"<think>.*?</think>\s*", "", raw_reflection, flags=re.DOTALL).strip()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(reflection)

    stripped = len(raw_reflection) - len(reflection)
    print(f"\nReflection saved to: {output_path}")
    print(f"Length: {len(reflection)} chars (stripped {stripped} chars of <think> reasoning)")
    print("\n--- Full Reflection ---")
    print(reflection)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play-Reflect-Transfer Pipeline")
    parser.add_argument(
        "--log-dir",
        action="append",
        default=[],
        help="Directory containing game logs (repeatable for multi-game). Supports globs like 'logs/reflect/*_r1'",
    )
    parser.add_argument("--model", required=True, help="Model to use for reflection")
    parser.add_argument(
        "--output", required=True, help="Output path for reflection text"
    )
    parser.add_argument(
        "--log-pattern",
        default="*.json",
        help="Glob pattern for log files within each log-dir (default: *.json)",
    )
    parser.add_argument(
        "--max-per-dir",
        type=int,
        default=0,
        help="Max logs per directory (0 = no limit). Use to cap token usage.",
    )
    parser.add_argument(
        "--prior-reflection",
        type=str,
        default="",
        help="Path to prior reflection file to iterate on (for r2, r3, etc.)",
    )
    parser.add_argument(
        "--placebo",
        action="store_true",
        help="Generate a placebo-control reflection: skip trajectory loading, prompt the model "
             "to write a 1200-word playbook from general knowledge with no game logs as input.",
    )

    args = parser.parse_args()

    if args.placebo:
        prompt = build_placebo_prompt()
        kwargs: dict = {"model": args.model, "messages": [{"role": "user", "content": prompt}]}
        if args.model.startswith("custom"):
            kwargs.update({
                "model": args.model.split("@")[0].replace("custom/", "openai/"),
                "base_url": args.model.split("@")[1],
                "api_key": os.environ.get("CUSTOM_API_KEY", "EMPTY"),
            })
        raw = completion(**kwargs).choices[0].message.content
        reflection = re.sub(r"<think>.*?</think>\s*", "", raw, flags=re.DOTALL).strip()
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            f.write(reflection)
        print(f"Placebo reflection saved to: {args.output} ({len(reflection)} chars)")
        print("\n--- Full Reflection ---")
        print(reflection)
        raise SystemExit(0)

    # Expand globs in --log-dir
    expanded_dirs = []
    for d in args.log_dir:
        matches = sorted(glob.glob(d))
        dirs = [m for m in matches if os.path.isdir(m)]
        if dirs:
            expanded_dirs.extend(dirs)
        elif os.path.isdir(d):
            expanded_dirs.append(d)
        else:
            print(f"Warning: no directories matching '{d}'")

    if not expanded_dirs:
        parser.error("No valid log directories found. Use --log-dir with a path or glob pattern.")

    asyncio.run(
        run_reflection(
            expanded_dirs, args.model, args.output, args.log_pattern,
            args.max_per_dir, args.prior_reflection
        )
    )
