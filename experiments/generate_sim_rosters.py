import random
import os
import sys
import itertools
import json
import argparse
import glob
import math

# Add project root to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.utils import load_roster_template

NAME_POOL = [
    "James",
    "Mary",
    "Robert",
    "Patricia",
    "John",
    "Jennifer",
    "Michael",
    "Linda",
    "David",
    "Elizabeth",
    "William",
    "Barbara",
    "Richard",
    "Susan",
    "Joseph",
    "Jessica",
    "Thomas",
    "Sarah",
    "Charles",
    "Karen",
    "Christopher",
    "Nancy",
    "Daniel",
    "Lisa",
    "Matthew",
    "Betty",
    "Anthony",
    "Margaret",
    "Mark",
    "Sandra",
    "Donald",
    "Ashley",
    "Steven",
    "Kimberly",
    "Paul",
    "Emily",
    "Andrew",
    "Donna",
    "Joshua",
    "Michelle",
    "Kenneth",
    "Dorothy",
    "Kevin",
    "Carol",
    "Brian",
    "Amanda",
    "George",
    "Melissa",
    "Edward",
    "Deborah",
    "Ronald",
    "Stephanie",
    "Timothy",
    "Rebecca",
    "Jason",
    "Sharon",
    "Jeffrey",
    "Laura",
    "Ryan",
    "Cynthia",
    "Jacob",
    "Kathleen",
    "Gary",
    "Amy",
    "Nicholas",
    "Shirley",
    "Eric",
    "Angela",
    "Jonathan",
    "Helen",
    "Stephen",
    "Anna",
    "Larry",
    "Brenda",
    "Justin",
    "Pamela",
    "Scott",
    "Nicole",
    "Brandon",
    "Emma",
]


def _check_unseeded_rosters(roster_output_dir: str) -> bool:
    """Check if existing rosters lack a seed. Returns True if safe to proceed."""
    existing = glob.glob(os.path.join(roster_output_dir, "*.json"))
    unseeded = []
    for f in existing:
        try:
            with open(f) as fh:
                data = json.load(fh)
            if "seed" not in data:
                unseeded.append(os.path.basename(f))
        except Exception:
            continue
    if unseeded:
        print(
            f"  WARNING: {len(unseeded)} existing rosters have no seed (not reproducible)."
        )
        print(f"  Examples: {unseeded[:3]}")
        resp = input("  Overwrite unseeded rosters? [y/N]: ").strip().lower()
        return resp == "y"
    return True


def _make_roster(current_config: dict, roster_seed: int) -> None:
    """Apply seed, shuffle agents/names, store seed in config."""
    random.seed(roster_seed)
    agents = current_config["agents"]
    random.shuffle(agents)
    name_pool = NAME_POOL.copy()
    random.shuffle(name_pool)
    for idx, agent in enumerate(agents):
        agent["name"] = name_pool[idx]
    current_config["seed"] = roster_seed


def generate_rosters(
    game_names: list[str],
    models: list[str],
    per_model_target: int = 30,
    overwrite: bool = False,
    challenger: str | None = None,
    roster_dir: str = "",
    seed: int = 42,
    reflection_file: str = "",
    reflect_team: str = "",
    reflect_model: str = "",
    no_swap: bool = False,
) -> None:
    """
    Generate roster files for ELO tournament.

    Roster count per game is scaled so each model gets approximately
    per_model_target player-slots in that game:
      individual games: n_rosters = ceil(target * n_models / n_players)
      team games: team_episodes per pair = ceil(target / (2 * (n_models - 1)))

    Uses a deterministic seed for reproducibility. Each roster gets
    seed = base_seed + roster_index so results are deterministic but distinct.

    For self-play (1 model): all agents use the same model.
    For team games with 2 models: even split — half with model A as team1, half swapped.
    For reflection: --reflection-file and --reflect-team inject reflection into one team.
    """

    if not isinstance(game_names, list):
        game_names = [game_names]

    print(f"Generating rosters for games: {game_names}")
    print(f"Competitors: {models}")
    print(f"Seed: {seed}")
    print(f"Per-model target: ~{per_model_target} player-slots per model per game")
    print("\n" + "=" * 50)

    for game_name in game_names:
        print(f"Processing game: {game_name}")

        try:
            template = load_roster_template(game_name)
        except Exception as e:
            print(f"  Error loading template for '{game_name}': {e}")
            continue

        if len(models) < 1:
            print(f"  Skipping {game_name}: Need at least 1 model.")
            continue

        roster_output_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "rosters", roster_dir, game_name)
            if roster_dir
            else os.path.join(os.path.dirname(__file__), "rosters", game_name)
        )
        os.makedirs(roster_output_dir, exist_ok=True)
        print(f"  Output directory: {roster_output_dir}")

        # Detect game type by checking for 2 distinct teams
        sample_agents = template["agents"]
        unique_teams = sorted({a.get("team") for a in sample_agents if a.get("team")})
        is_team_game = len(unique_teams) == 2

        # Scale per-game count so each model gets ~per_model_target player-slots.
        n_players = len(sample_agents)
        n_models = len(models)
        if is_team_game and n_models >= 2:
            effective_team_episodes_override = max(
                1, math.ceil(per_model_target / (2 * (n_models - 1)))
            )
            effective_individual_scenarios = (
                effective_team_episodes_override * n_models * (n_models - 1)
            )
        else:
            effective_team_episodes_override = None
            effective_individual_scenarios = max(
                1, math.ceil(per_model_target * n_models / max(1, n_players))
            )

        # Check for unseeded existing rosters if overwriting
        if overwrite and not _check_unseeded_rosters(roster_output_dir):
            print(f"  Skipping {game_name} (user declined overwrite)")
            continue

        # Global roster counter for seed offset
        roster_counter = 0
        count = 0

        if is_team_game:
            # --- TEAM GAME: pairwise, team_episodes per pair ---
            print(f"  Type: team-based ({unique_teams[0]} vs {unique_teams[1]})")
            t1, t2 = unique_teams[0], unique_teams[1]

            # Self-play: single model on both teams
            if len(models) == 1:
                pairs = [(models[0], models[0])]
                team_episodes = effective_individual_scenarios
            elif len(models) == 2 and not no_swap:
                # Even split: half with A as t1, half with B as t1
                half = max(1, effective_individual_scenarios // 2)
                pairs = [(models[0], models[1]), (models[1], models[0])]
                team_episodes = half
                print(f"  Even split: {half} eps per side")
            elif len(models) == 2 and no_swap:
                # Fixed assignment: models[0] always on t1, models[1] always on t2
                pairs = [(models[0], models[1])]
                team_episodes = effective_individual_scenarios
                print(f"  Fixed assignment: {models[0]} as {t1}, {models[1]} as {t2}")
            else:
                pairs = list(itertools.permutations(models, 2))
                team_episodes = effective_team_episodes_override or 1
                print(f"  {len(pairs)} pairs × {team_episodes} eps each")

            for pair_idx, (model_a, model_b) in enumerate(pairs):
                if challenger and challenger not in (model_a, model_b):
                    continue
                for i in range(team_episodes):
                    filename = (
                        f"roster_{game_name}_{roster_dir}_ep{roster_counter}.json"
                    )
                    file_path = os.path.join(roster_output_dir, filename)
                    if not overwrite and os.path.exists(file_path):
                        roster_counter += 1
                        continue

                    try:
                        current_config = load_roster_template(game_name)
                    except Exception as e:
                        print(f"  Error loading template: {e}")
                        continue

                    agents = current_config["agents"]
                    for agent in agents:
                        if agent.get("team") == t1:
                            agent["agent_model"] = model_a
                        elif agent.get("team") == t2:
                            agent["agent_model"] = model_b
                        else:
                            agent["agent_model"] = model_a

                    # Reflection: tag agents whose model matches reflect_model AND team matches reflect_team
                    if reflection_file and (reflect_team or reflect_model):
                        current_config["reflection_file"] = os.path.abspath(
                            reflection_file
                        )
                        for agent in agents:
                            team_match = (
                                (agent.get("team") == reflect_team)
                                if reflect_team
                                else True
                            )
                            model_match = (
                                (reflect_model in agent.get("agent_model", ""))
                                if reflect_model
                                else True
                            )
                            agent["include_reflection"] = team_match and model_match

                    _make_roster(current_config, seed + roster_counter)
                    roster_counter += 1

                    with open(file_path, "w") as f:
                        json.dump(current_config, f, indent=4)
                    count += 1

        elif len(sample_agents) == 2:
            # --- 2-PLAYER INDIVIDUAL GAME: pairwise matchups ---
            pairs = list(itertools.combinations(models, 2))
            pair_episodes = max(
                1, math.ceil(effective_individual_scenarios / len(pairs))
            )
            print(
                f"  Type: individual (2 players), {len(pairs)} pairs × {pair_episodes} eps"
            )
            for pair_idx, (model_a, model_b) in enumerate(pairs):
                if challenger and challenger not in (model_a, model_b):
                    continue
                for i in range(pair_episodes):
                    filename = (
                        f"roster_{game_name}_{roster_dir}_ep{roster_counter}.json"
                    )
                    file_path = os.path.join(roster_output_dir, filename)
                    if not overwrite and os.path.exists(file_path):
                        roster_counter += 1
                        continue

                    try:
                        current_config = load_roster_template(game_name)
                    except Exception as e:
                        print(f"  Error loading template: {e}")
                        continue

                    agents = current_config["agents"]
                    agents[0]["agent_model"] = model_a
                    agents[1]["agent_model"] = model_b

                    _make_roster(current_config, seed + roster_counter)
                    roster_counter += 1

                    with open(file_path, "w") as f:
                        json.dump(current_config, f, indent=4)
                    count += 1

        else:
            # --- MULTI-PLAYER INDIVIDUAL GAME: random model per player slot ---
            print(
                f"  Type: individual ({len(sample_agents)} players, "
                f"{effective_individual_scenarios} rosters)"
            )
            for i in range(effective_individual_scenarios):
                filename = f"roster_{game_name}_{roster_dir}_ep{roster_counter}.json"
                file_path = os.path.join(roster_output_dir, filename)
                if not overwrite and os.path.exists(file_path):
                    roster_counter += 1
                    continue

                try:
                    current_config = load_roster_template(game_name)
                except Exception as e:
                    print(f"  Error loading template: {e}")
                    continue

                agents = current_config["agents"]
                n_agents = len(agents)

                # Seed for model assignment (before _make_roster shuffles)
                random.seed(seed + roster_counter)

                # Assign random model to each player slot
                if n_agents <= len(models):
                    assigned = random.sample(models, n_agents)
                else:
                    assigned = [random.choice(models) for _ in range(n_agents)]

                # Challenger filter: skip if challenger not in assigned models
                if challenger and challenger not in assigned:
                    roster_counter += 1
                    continue

                for agent, model in zip(agents, assigned):
                    agent["agent_model"] = model

                _make_roster(current_config, seed + roster_counter)
                roster_counter += 1

                with open(file_path, "w") as f:
                    json.dump(current_config, f, indent=4)
                count += 1

        print(f"  Generated {count} NEW rosters for {game_name}")

    print("\nGeneration Complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Rosters for ELO Tournament")
    parser.add_argument(
        "--game",
        nargs="+",
        default=[
            "rock_paper_scissors",
            "battle_of_the_sexes",
            "chicken",
            "stag_hunt",
            "centipede",
            "prisoners_dilemma",
            "minority_game",
            "public_goods",
            "bargaining",
            "dead_last",
            "undercover",
            "spyfall",
            "chameleon",
            "insider",
            "liars_dice",
            "werewolves",
            "resistance",
            "coup",
            "survivor",
            "sheriff",
            "skull",
        ],
        help="List of games",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "gpt-4o-mini",
            "gpt-4o",
            "custom/Qwen/Qwen3-32B@http://localhost:8002/v1",
            "custom/google/gemma-3-27b-it@http://localhost:8003/v1",
            "custom/Qwen/Qwen3-4B-Instruct-2507@http://localhost:8001/v1",
            "custom/Qwen/Qwen2.5-3B-Instruct@http://localhost:8000/v1",
        ],
        help="List of models to compete",
    )
    parser.add_argument(
        "--per-model-target",
        type=int,
        default=30,
        help="Target player-slots per model per game. Roster count is scaled per-game based on player count so each model appears ~this many times (default: 30).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow generation even if output directory is not empty",
    )
    parser.add_argument(
        "--challenger",
        type=str,
        default=None,
        help="If set, only generate rosters involving this model",
    )
    parser.add_argument(
        "--roster-dir",
        type=str,
        default="v2",
        help="Subdirectory under experiments/rosters/ for this run (default: v2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible roster generation (default: 42)",
    )
    parser.add_argument(
        "--reflection-file",
        type=str,
        default="",
        help="Path to reflection text file to inject into agent prompts",
    )
    parser.add_argument(
        "--reflect-team",
        type=str,
        default="",
        help="Team name that receives the reflection (e.g., 'Werewolves', 'Villagers')",
    )
    parser.add_argument(
        "--reflect-model",
        type=str,
        default="",
        help="Model substring that receives reflection (e.g., 'Qwen', 'gpt-5'). Combined with --reflect-team if both set.",
    )
    parser.add_argument(
        "--no-swap",
        action="store_true",
        help="For 2-model team games: don't swap sides. models[0] always plays team 1, models[1] team 2.",
    )

    args = parser.parse_args()

    generate_rosters(
        game_names=args.game,
        models=args.models,
        per_model_target=args.per_model_target,
        overwrite=args.overwrite,
        challenger=args.challenger,
        roster_dir=args.roster_dir,
        seed=args.seed,
        reflection_file=args.reflection_file,
        reflect_team=args.reflect_team,
        reflect_model=args.reflect_model,
        no_swap=args.no_swap,
    )
