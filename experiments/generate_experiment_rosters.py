"""
Generate experiment rosters with explicit model-to-team assignment.

Usage:
    # Self-play, no reflection
    python -m experiments.generate_experiment_rosters \
        --game werewolves --n 30 --roster-dir selfplay_gpt5 \
        --all-model gpt-5

    # Self-play, all agents get reflection
    python -m experiments.generate_experiment_rosters \
        --game werewolves --n 10 \
        --roster-dir reflect/werewolf_r2/baselines/selfplay_gpt5_r1 \
        --all-model gpt-5 \
        --team Werewolves --reflection experiments/reflections/gpt5_werewolf_r1.txt \
        --team Villagers --reflection experiments/reflections/gpt5_werewolf_r1.txt

    # Cross-model, Qwen gets reflection
    python -m experiments.generate_experiment_rosters \
        --game werewolves --n 30 --roster-dir qwen32b_R_wolf_vs_gpt5 \
        --team Werewolves --model "custom/Qwen/..." \
        --team Villagers --model gpt-5 \
        --reflection experiments/reflections/gpt5_werewolf_r1.txt --reflect-model Qwen
"""

import argparse
import json
import os
import random
import sys
import glob

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.utils import load_roster_template

NAME_POOL = [
    "James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael", "Linda",
    "David", "Elizabeth", "William", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
    "Thomas", "Sarah", "Charles", "Karen", "Christopher", "Nancy", "Daniel", "Lisa",
    "Matthew", "Betty", "Anthony", "Margaret", "Mark", "Sandra", "Donald", "Ashley",
    "Steven", "Kimberly", "Paul", "Emily", "Andrew", "Donna", "Joshua", "Michelle",
    "Kenneth", "Dorothy", "Kevin", "Carol", "Brian", "Amanda", "George", "Melissa",
    "Edward", "Deborah", "Ronald", "Stephanie", "Timothy", "Rebecca", "Jason", "Sharon",
    "Jeffrey", "Laura", "Ryan", "Cynthia", "Jacob", "Kathleen", "Gary", "Amy",
    "Nicholas", "Shirley", "Eric", "Angela", "Jonathan", "Helen", "Stephen", "Anna",
    "Larry", "Brenda", "Justin", "Pamela", "Scott", "Nicole", "Brandon", "Emma",
]


def generate_experiment_rosters(
    game: str,
    n: int,
    roster_dir: str,
    team_configs: list[dict],
    all_model: str = "",
    seed: int = 42,
    overwrite: bool = False,
    reflection: str = "",
    reflect_model: str = "",
    reflect_team: str = "",
) -> None:
    """
    Generate N rosters with explicit model/reflection assignment.

    reflection: path to reflection file
    reflect_model: model substring that receives the reflection (e.g., 'Qwen', 'gpt-5')
    """
    load_roster_template(game)  # validate game exists

    out_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "rosters", roster_dir, game)
    )
    os.makedirs(out_dir, exist_ok=True)

    # Count existing rosters to support extending
    existing = len(glob.glob(os.path.join(out_dir, "*.json")))
    if existing > 0 and not overwrite:
        print(f"  {existing} rosters already exist. Generating {max(0, n - existing)} more.")
        start = existing
    else:
        start = 0
        if overwrite and existing > 0:
            for f in glob.glob(os.path.join(out_dir, "*.json")):
                os.remove(f)
            print(f"  Removed {existing} existing rosters.")

    # Build team -> config lookup
    team_lookup = {tc["team"]: tc for tc in team_configs}

    # Check reflection file exists
    if reflection:
        abs_rf = os.path.abspath(reflection)
        if not os.path.exists(abs_rf):
            raise FileNotFoundError(f"Reflection file not found: {abs_rf}")
        if not reflect_model and not reflect_team:
            raise ValueError("--reflect-model or --reflect-team is required when --reflection is set")
        if reflect_model and reflect_team:
            raise ValueError("Specify only one of --reflect-model or --reflect-team, not both")
        # Warn if all agents use the same model and reflect_model is specified
        all_models = set()
        if all_model:
            all_models.add(all_model)
        else:
            all_models = {tc.get("model", "") for tc in team_configs}
        if reflect_model and len(all_models) == 1 and any(reflect_model in m for m in all_models):
            print(f"  WARNING: All agents use the same model matching '{reflect_model}'. "
                  f"All agents will receive reflection. Use --reflect-team instead to target one team.")

    dir_label = os.path.basename(roster_dir.rstrip("/"))
    count = 0
    for i in range(start, n):
        random.seed(seed + i)

        config = load_roster_template(game)
        agents = config["agents"]

        has_reflection = False
        for agent in agents:
            team = agent.get("team", "")
            tc = team_lookup.get(team)

            # Model assignment
            if all_model:
                agent["agent_model"] = all_model
            elif tc and tc.get("model"):
                agent["agent_model"] = tc["model"]

            # Reflection assignment: match by model substring AND optionally team
            model_match = (not reflect_model) or (reflect_model in agent.get("agent_model", ""))
            team_match = (not reflect_team) or (agent.get("team", "") == reflect_team)
            if reflection and model_match and team_match:
                agent["include_reflection"] = True
                has_reflection = True
            else:
                agent["include_reflection"] = False

        # Set reflection file path
        if has_reflection and reflection:
            config["reflection_file"] = os.path.abspath(reflection)

        # Shuffle agents and assign names
        random.shuffle(agents)
        name_pool = NAME_POOL.copy()
        random.shuffle(name_pool)
        for idx, agent in enumerate(agents):
            agent["name"] = name_pool[idx]

        config["seed"] = seed + i

        filename = f"roster_{game}_{dir_label}_ep{i}.json"
        filepath = os.path.join(out_dir, filename)
        with open(filepath, "w") as f:
            json.dump(config, f, indent=4)
        count += 1

    print(f"  Generated {count} rosters in {out_dir}")
    print(f"  Total: {len(glob.glob(os.path.join(out_dir, '*.json')))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate experiment rosters with explicit model-to-team assignment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--game", required=True, help="Game name (e.g., werewolves)")
    parser.add_argument("--n", type=int, required=True, help="Number of rosters to generate")
    parser.add_argument("--roster-dir", required=True, help="Subdirectory under experiments/rosters/")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing rosters")

    # All-agents model
    parser.add_argument("--all-model", type=str, default="", help="All agents use this model")

    # Per-team config
    parser.add_argument("--team", action="append", default=[], help="Team name (repeatable)")
    parser.add_argument("--model", action="append", default=[], help="Model for the corresponding --team (repeatable)")

    # Reflection
    parser.add_argument("--reflection", type=str, default="", help="Reflection file path")
    parser.add_argument("--reflect-model", type=str, default="", help="Model substring that receives the reflection (use when models differ between teams)")
    parser.add_argument("--reflect-team", type=str, default="", help="Team that receives the reflection (use when both teams have the same model)")

    args = parser.parse_args()

    # Build team configs from repeated args
    team_configs = []
    for i in range(len(args.team)):
        tc = {
            "team": args.team[i],
            "model": args.model[i] if i < len(args.model) else "",
        }
        team_configs.append(tc)

    if not args.all_model and not team_configs:
        parser.error("Must specify --all-model or at least one --team/--model pair")

    generate_experiment_rosters(
        game=args.game,
        n=args.n,
        roster_dir=args.roster_dir,
        team_configs=team_configs,
        all_model=args.all_model,
        seed=args.seed,
        overwrite=args.overwrite,
        reflection=args.reflection,
        reflect_model=args.reflect_model,
        reflect_team=args.reflect_team,
    )
