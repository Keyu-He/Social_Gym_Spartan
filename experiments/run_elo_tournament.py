import asyncio
import logging
import os
import sys
import json
import glob
from datetime import datetime
from tqdm.asyncio import tqdm

# Add project root to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sotopia.server import arun_one_episode
from experiments.utils import get_game_module, load_game_config

# Logger for this script
logger = logging.getLogger(__name__)


async def run_elo_tournament(
    game_names: list[str],
    tag: str = "elo_v2",
    push_to_db: bool = True,
    concurrency_limit: int = 10,
    roster_dir: str = "",
    log_dir: str = "logs",
) -> None:
    """
    Run ELO tournament by executing pre-generated rosters found in experiments/rosters/.
    """
    if not isinstance(game_names, list):
        game_names = [game_names]

    print("Starting ELO Tournament Execution")
    print(f"Target Games: {game_names}")
    print(f"Concurrency: {concurrency_limit}")
    print("\n" + "=" * 50)

    semaphore = asyncio.Semaphore(concurrency_limit)

    # 1. Scan executed rosters from existing logs
    # Key: "roster_dir/roster_file" for uniqueness across experiments
    executed_rosters = set()
    existing_logs = glob.glob(os.path.join(log_dir, "**", "*.json"), recursive=True)
    for log_file in existing_logs:
        try:
            with open(log_file, "r") as f:
                data = json.load(f)
                meta = data.get("metadata", {})
                r_file = meta.get("roster_file", "")
                r_dir = meta.get("roster_dir", "")
                if r_file:
                    executed_rosters.add(f"{r_dir}/{r_file}" if r_dir else r_file)
        except Exception as e:
            print(f"Error reading log file {log_file}: {e}")
    print(f"Found {len(executed_rosters)} already executed rosters.")

    # 2. Collect all rosters across all games
    all_rosters: list[tuple[str, str]] = []  # (game_name, roster_path)
    game_modules: dict[str, any] = {}

    for game_name in game_names:
        roster_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "rosters", roster_dir, game_name)
            if roster_dir
            else os.path.join(os.path.dirname(__file__), "rosters", game_name)
        )
        roster_files = sorted(glob.glob(os.path.join(roster_path, "*.json")))

        remaining = [
            r
            for r in roster_files
            if f"{roster_dir}/{os.path.basename(r)}" not in executed_rosters
        ]
        print(f"  {game_name}: {len(remaining)}/{len(roster_files)} rosters to run")

        if not remaining:
            continue

        try:
            game_modules[game_name] = get_game_module(game_name)
        except Exception as e:
            print(f"  Error loading game module for {game_name}: {e}")
            continue

        for r in remaining:
            all_rosters.append((game_name, r))

    print(f"\nTotal rosters to run: {len(all_rosters)}")

    if not all_rosters:
        print("Nothing to do.")
        return

    # 3. Define worker that handles any game
    async def _worker(game_name: str, roster_path: str) -> None:
        async with semaphore:
            filename = os.path.basename(roster_path)
            try:
                with open(roster_path, "r") as f:
                    roster_config = json.load(f)

                base_config = load_game_config(game_name)
                episode_config = base_config.copy()
                episode_config.update(roster_config)

                agent_model_list = [a["agent_model"] for a in episode_config["agents"]]
                agents_conf = episode_config["agents"]
                teams = set(a.get("team") for a in agents_conf if a.get("team"))

                unique_models = sorted(list(set(agent_model_list)))
                model_a_log = unique_models[0] if len(unique_models) > 0 else "unknown"
                model_b_log = (
                    unique_models[1] if len(unique_models) > 1 else model_a_log
                )

                metadata = {
                    "game_name": game_name,
                    "model_a": model_a_log,
                    "model_b": model_b_log,
                    "roster_file": filename,
                    "roster_dir": roster_dir,
                }

                if len(teams) > 1:
                    for team_name in teams:
                        team_models = set(
                            a["agent_model"]
                            for a in agents_conf
                            if a.get("team") == team_name
                        )
                        if len(team_models) == 1:
                            metadata[f"{team_name}_model"] = list(team_models)[0]
                        else:
                            metadata[f"{team_name}_model"] = "mixed"

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                log_filename = f"episode_{tag}_{game_name}_{timestamp}.json"
                log_path = os.path.join(log_dir, tag, log_filename)

                prepare_scenario = game_modules[game_name].prepare_scenario
                env, agents = prepare_scenario(
                    env_model_name="gpt-4o",
                    agent_model_name=agent_model_list,
                    config=episode_config,
                )

                os.makedirs(os.path.join(log_dir, tag), exist_ok=True)

                await arun_one_episode(
                    env=env,
                    agent_list=agents,
                    tag=tag,
                    push_to_db=push_to_db,
                    output_path=log_path,
                    metadata=metadata,
                )
            except Exception as e:
                import traceback

                print(f"\nERROR in {filename}: {e}\n{traceback.format_exc()}")

    # 4. Launch all rosters across all games concurrently
    tasks = [
        asyncio.create_task(_worker(game_name, r_path))
        for game_name, r_path in all_rosters
    ]
    print(f"Queuing {len(tasks)} tasks (concurrency={concurrency_limit})...")
    for f in tqdm(
        asyncio.as_completed(tasks), total=len(tasks), desc="Playing all games"
    ):
        await f

    print("\nAll Scheduled Rosters Executed.")


if __name__ == "__main__":
    import argparse

    # Reconfigure logging to suppress sotopia's verbose output
    # 1. Root Logger
    logging.basicConfig(level=logging.ERROR)

    # 2. Silence sotopia experimental server and generation
    server_logger = logging.getLogger("sotopia.experimental.server")
    server_logger.setLevel(logging.ERROR)
    generation_logger = logging.getLogger("sotopia.generation")
    generation_logger.setLevel(logging.ERROR)

    os.environ.setdefault("REDIS_OM_URL", "redis://:@localhost:6379")

    parser = argparse.ArgumentParser(description="Run ELO Tournament Execution Phase")
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
        help="List of games to execute rosters for",
    )
    parser.add_argument("--tag", type=str, default="elo_v2", help="Experiment tag")
    parser.add_argument(
        "--concurrency", type=int, default=10, help="Max concurrent episodes"
    )
    parser.add_argument(
        "--roster-dir",
        type=str,
        default="tournament",
        help="Subdirectory under experiments/rosters/ for this run (default: tournament)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs/tournament",
        help="Directory to write game logs (default: logs/tournament)",
    )

    args = parser.parse_args()

    asyncio.run(
        run_elo_tournament(
            game_names=args.game,
            tag=args.tag,
            concurrency_limit=args.concurrency,
            roster_dir=args.roster_dir,
            log_dir=args.log_dir,
        )
    )
