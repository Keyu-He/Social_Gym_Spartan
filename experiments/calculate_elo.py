import csv
import glob
import json
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression

from experiments.utils import load_game_config

# ---------------------------------------------------------------------------
# Rating constants
# ---------------------------------------------------------------------------
K_FACTOR = 32  # online ELO step size
STARTING_ELO = 1000  # anchor rating for both methods (Chatbot Arena uses 1000)
ELO_EPOCHS = 1  # passes over the data for the online ELO method

# Games where the optimal outcome requires cooperation / coordination.
# These are excluded from the main competitive ELO and shown separately.
COOPERATIVE_GAMES = {"battle_of_the_sexes", "stag_hunt", "public_goods", "centipede"}

# Canonical model name mapping: raw model string → display name.
# Handles endpoint changes, naming inconsistencies, etc.
MODEL_NAME_MAP = {
    # OpenAI
    "gpt-4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-5": "gpt-5",
    # Qwen 3 32B (includes old qwen3.5-35b-a3b which was replaced)
    "qwen/qwen3-32b": "Qwen3-32B",
    "Qwen/Qwen3-32B": "Qwen3-32B",
    "qwen/qwen3.5-35b-a3b": "Qwen3-32B",
    # Qwen 3 4B
    "qwen/qwen3-4b-2507": "Qwen3-4B",
    "Qwen/Qwen3-4B-Instruct-2507": "Qwen3-4B",
    # Qwen 2.5 3B
    "qwen2.5-3b-instruct": "Qwen2.5-3B",
    "Qwen/Qwen2.5-3B-Instruct": "Qwen2.5-3B",
    # Gemma
    "google/gemma-3-27b": "Gemma3-27B",
    "google/gemma-3-27b-it": "Gemma3-27B",
}


def normalize_model_name(raw: str) -> str:
    """Normalize a raw model string to a canonical display name."""
    # Strip custom/ prefix and @url suffix
    name = raw.split("@")[0].replace("custom/", "")
    if name in MODEL_NAME_MAP:
        return MODEL_NAME_MAP[name]
    return name


def bootstrap_mle_elo(
    win_counts: Dict[tuple, int],
    tie_counts: Dict[tuple, int],
    models: list[str],
    n_bootstrap: int = 1000,
    seed: int = 42,
    **mle_kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bootstrap CI for the BT MLE via multinomial resampling on aggregated counts.

    Returns (point, ci_lo, ci_hi) arrays aligned with `models`. Mirrors the
    `compute_bootstrap_bt` approach in FastChat: each iteration draws N total
    battles with replacement (multinomial) from the empirical distribution
    over unique (winner, loser) and (tied-pair) cells, refits BT.
    """
    point = compute_mle_elo(win_counts, tie_counts, models, **mle_kwargs)
    if not win_counts and not tie_counts:
        return point, point.copy(), point.copy()

    rng = np.random.default_rng(seed)
    n_models = len(models)
    samples = np.zeros((n_bootstrap, n_models))

    win_keys = list(win_counts.keys())
    tie_keys = list(tie_counts.keys())
    counts = np.array(
        [win_counts[k] for k in win_keys] + [tie_counts[k] for k in tie_keys],
        dtype=float,
    )
    total = int(counts.sum())
    if total == 0:
        return point, point.copy(), point.copy()
    probs = counts / counts.sum()
    n_win = len(win_keys)

    for b in range(n_bootstrap):
        draw = rng.multinomial(total, probs)
        wc = {k: int(v) for k, v in zip(win_keys, draw[:n_win]) if v > 0}
        tc = {k: int(v) for k, v in zip(tie_keys, draw[n_win:]) if v > 0}
        try:
            samples[b] = compute_mle_elo(wc, tc, models, **mle_kwargs)
        except Exception:
            samples[b] = point  # fallback if a draw produces a degenerate fit

    ci_lo = np.percentile(samples, 2.5, axis=0)
    ci_hi = np.percentile(samples, 97.5, axis=0)
    return point, ci_lo, ci_hi


def compute_mle_elo(
    win_counts: Dict[tuple, int],
    tie_counts: Dict[tuple, int],
    models: list[str],
    scale: float = 400.0,
    base: float = 10.0,
    init_rating: float = float(STARTING_ELO),
    C: float = 0.1,
) -> np.ndarray:
    """Chatbot Arena's MLE-based Elo via L2-regularized logistic regression.

    Implements the FastChat / LMSYS Chatbot Arena rating method:
      - feature matrix X has +log(base) in the model_a column and -log(base)
        in the model_b column for each battle.
      - target Y = 1 if model_a won, 0 if model_b won.
      - ties are duplicated as (a-wins=1, b-wins=0) — both rows are added so
        a tie contributes one win and one loss to each side.
      - sklearn LogisticRegression(fit_intercept=False, penalty='l2') fits
        coefficients beta_i; the resulting Elo is `scale * beta_i + init_rating`.
      - L2 strength: C=0.1 by default. Stronger than sklearn's C=1.0 default
        because our per-game subsets are small (~30 pairwise comparisons per
        model) and exhibit occasional complete separation (e.g., a model
        winning 30-0 in Chicken) where the unregularized MLE would diverge.
        At our scale, C=1.0 lets sweep cases saturate the rating range while
        C=0.1 tames them without washing out signal in normal games.

    win_counts[(a, b)] = times model a beat model b (no ties).
    tie_counts[(a, b)] = times model a and b tied (unordered; only one of
        (a,b) or (b,a) should be populated per pair).
    models: ordered list of model names; the returned array matches this order.

    Returns: 1D numpy array of Elo ratings, indexed by `models`.
    """
    p = len(models)
    if p == 0:
        return np.zeros(0)
    idx = {m: i for i, m in enumerate(models)}

    # Canonicalize each battle to (a, b) with a < b alphabetically, then
    # set Y=1 if a won, Y=0 if b won. This matches the FastChat / Chatbot
    # Arena encoding and keeps both Y=0 and Y=1 classes present.
    rows: list[tuple[int, int, float, float]] = []
    for (winner, loser), c in win_counts.items():
        if c <= 0 or winner not in idx or loser not in idx:
            continue
        a, b = sorted((winner, loser))
        y = 1.0 if winner == a else 0.0
        rows.append((idx[a], idx[b], y, float(c)))
    for (a_raw, b_raw), c in tie_counts.items():
        if c <= 0 or a_raw not in idx or b_raw not in idx:
            continue
        a, b = sorted((a_raw, b_raw))
        # Duplicate: one row counts as a-wins, one as b-wins.
        rows.append((idx[a], idx[b], 1.0, float(c)))
        rows.append((idx[a], idx[b], 0.0, float(c)))

    if not rows:
        return np.full(p, init_rating)

    # If only one class is present (extreme legacy / no-tie case), inject
    # a tiny pseudo-row for the missing class so sklearn can fit at all.
    has_y0 = any(y == 0.0 for _, _, y, _ in rows)
    has_y1 = any(y == 1.0 for _, _, y, _ in rows)
    if not (has_y0 and has_y1):
        # Add a near-zero-weight balancing row using the first canonical pair.
        ia, ib, _, _ = rows[0]
        rows.append((ia, ib, 0.0 if has_y1 else 1.0, 1e-6))

    n = len(rows)
    X = np.zeros((n, p))
    Y = np.zeros(n)
    W = np.zeros(n)
    log_base = math.log(base)
    for r, (ia, ib, y, w) in enumerate(rows):
        X[r, ia] = +log_base
        X[r, ib] = -log_base
        Y[r] = y
        W[r] = w

    lr = LogisticRegression(
        fit_intercept=False, penalty="l2", C=C, tol=1e-6, max_iter=1000
    )
    lr.fit(X, Y, sample_weight=W)
    coef = lr.coef_[0]
    elo = scale * coef + init_rating
    # Anchor the mean to init_rating so figures stay centered.
    elo = elo - elo.mean() + init_rating
    return elo


def is_team_game(game_name: str) -> bool:
    """A team game has 2+ teams and >2 players.

    Some games (Werewolves, Spyfall, Undercover) build their agent list
    dynamically at scenario time, so the static config has no agents/teams.
    Treat the known hidden-role games as team games regardless of static config.
    """
    KNOWN_TEAM_GAMES = {
        "werewolves",
        "resistance",
        "spyfall",
        "chameleon",
        "insider",
        "undercover",
    }
    if game_name in KNOWN_TEAM_GAMES:
        return True
    config = load_game_config(game_name)
    agents = config.get("agents", [])
    teams = {a.get("team") for a in agents if a.get("team")}
    return len(teams) >= 2 and len(agents) > 2


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


# ---------------------------------------------------------------------------
# Episode parsing (shared between the two rating methods)
# ---------------------------------------------------------------------------


@dataclass
class ParsedEpisode:
    """Pre-processed episode log ready to feed any pairwise rating method."""

    model_mapping: Dict[str, str]  # agent name -> normalized model name
    agent_rewards: Dict[str, float]  # agent name -> reward (zero-sum-ish)
    pairs: list[tuple[str, str, bool]]  # (agent_a, agent_b, is_cross_team)
    unique_models: set[str]  # distinct models in this episode


def _parse_episode(filepath: str) -> Optional[ParsedEpisode]:
    """Load an episode log and return its data in a form both rating methods
    can iterate over. Returns None for malformed logs (caller should skip).
    """
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
    except Exception:
        return None

    raw_mapping = data.get("model_mapping", {})
    model_mapping = {k: normalize_model_name(v) for k, v in raw_mapping.items()}
    rewards = data.get("rewards", [])
    metadata = data.get("metadata", {})

    if not model_mapping or not rewards:
        return None

    parsed_rewards: list[float] = []
    for r in rewards:
        if isinstance(r, (list, tuple)):
            parsed_rewards.append(float(r[0]))
        else:
            parsed_rewards.append(float(r))
    if len(model_mapping) != len(parsed_rewards):
        return None

    agents = list(model_mapping.keys())
    agent_rewards = {agents[i]: parsed_rewards[i] for i in range(len(agents))}

    # Team detection: metadata keys ending in "_model" beyond the generic
    # model_a / model_b identify per-team model assignments. If there are
    # >=2 such keys, this is a team game and we restrict pairs to cross-team.
    team_model_keys = {
        k: v
        for k, v in metadata.items()
        if k.endswith("_model") and k not in ("model_a", "model_b")
    }
    alt_agents: list[str] = []
    main_agents: list[str] = []
    is_team = False
    if len(team_model_keys) >= 2:
        team_groups: dict[str, list[str]] = {}
        for team_key, team_model in team_model_keys.items():
            # metadata stores raw model strings; normalize before comparing
            # against model_mapping (which is already normalized).
            team_model_norm = normalize_model_name(team_model)
            members = [a for a, m in model_mapping.items() if m == team_model_norm]
            if members:
                team_groups[team_key] = members
        if len(team_groups) >= 2:
            sorted_teams = sorted(team_groups.values(), key=len)
            alt_agents = sorted_teams[0]  # smaller team = minority/hidden role
            main_agents = sorted_teams[-1]  # larger team = majority role
            is_team = True

    if is_team:
        pairs = [(a, b, True) for a in alt_agents for b in main_agents]
    else:
        pairs = [
            (agents[i], agents[j], False)
            for i in range(len(agents))
            for j in range(i + 1, len(agents))
        ]

    return ParsedEpisode(
        model_mapping=model_mapping,
        agent_rewards=agent_rewards,
        pairs=pairs,
        unique_models=set(model_mapping.values()),
    )


def _as_dict(
    ratings: Dict[str, float] | np.ndarray, models: Iterable[str]
) -> Dict[str, float]:
    """Normalize ratings (either a dict or a model-aligned ndarray) to a dict."""
    if isinstance(ratings, np.ndarray):
        return {m: float(ratings[i]) for i, m in enumerate(models)}
    return {m: float(v) for m, v in ratings.items()}


def _build_stats(
    models: Iterable[str],
    elo: Dict[str, float] | np.ndarray,
    elo_alt: Dict[str, float] | np.ndarray,
    elo_main: Dict[str, float] | np.ndarray,
    wins: Dict[str, int],
    total_pairwise: Dict[str, int],
    episodes_played: Dict[str, int],
    ci_lo: Optional[np.ndarray] = None,
    ci_hi: Optional[np.ndarray] = None,
) -> list[dict[str, Any]]:
    """Build the per-model stats list shared by both rating methods.

    `models` defines the order used to interpret ndarray-typed ratings.
    Output is sorted by descending overall ELO. If `ci_lo`/`ci_hi` are
    provided (BT bootstrap), they're carried through as elo_lo / elo_hi.
    """
    model_list = list(models)
    elo_d = _as_dict(elo, model_list)
    alt_d = _as_dict(elo_alt, model_list)
    main_d = _as_dict(elo_main, model_list)
    lo_d = _as_dict(ci_lo, model_list) if ci_lo is not None else None
    hi_d = _as_dict(ci_hi, model_list) if ci_hi is not None else None

    ranked = sorted(elo_d.keys(), key=lambda m: -elo_d[m])
    stats = []
    for rank, m in enumerate(ranked, 1):
        n_pw = total_pairwise.get(m, 0)
        win_rate = (wins.get(m, 0) / n_pw * 100) if n_pw > 0 else 0.0
        row = {
            "rank": rank,
            "model": m,
            "elo": elo_d[m],
            "elo_w": alt_d.get(m, float(STARTING_ELO)),
            "elo_v": main_d.get(m, float(STARTING_ELO)),
            "win_rate": win_rate,
            "matches": episodes_played.get(m, 0),
        }
        if lo_d is not None and hi_d is not None:
            row["elo_lo"] = lo_d[m]
            row["elo_hi"] = hi_d[m]
        stats.append(row)
    return stats


def generate_single_table_html(
    title: str, stats: list[dict[str, Any]], show_split_elo: bool = True
) -> str:
    """Generates the HTML for a single leaderboard table."""
    rows_html = ""
    for item in stats:
        rank = item["rank"]
        rank_display = f"#{rank}"
        if rank == 1:
            rank_display = "🥇"
        if rank == 2:
            rank_display = "🥈"
        if rank == 3:
            rank_display = "🥉"

        name = item["model"]
        provider = "Unknown"
        # Heuristic for provider
        lower_name = name.lower()
        if "gpt" in lower_name:
            provider = "OpenAI"
        elif "qwen" in lower_name:
            provider = "Alibaba"
        elif "gemini" in lower_name or "google" in lower_name:
            provider = "Google"
        elif "claude" in lower_name:
            provider = "Anthropic"
        elif "llama" in lower_name:
            provider = "Meta"
        elif "mistral" in lower_name:
            provider = "Mistral"

        wr_val = item["win_rate"]
        wr_color = f"hsl({int(wr_val * 1.2)}, 70%, 40%)"

        split_elo_cells = ""
        if show_split_elo:
            split_elo_cells = f"""
            <td class="elo-split">{int(item['elo_w'])}</td>
            <td class="elo-split">{int(item['elo_v'])}</td>
            """
        else:
            split_elo_cells = """
            <td class="elo-split" style="color: #ccc;">-</td>
            <td class="elo-split" style="color: #ccc;">-</td>
            """

        row = f"""
        <tr>
            <td class="rank">{rank_display}</td>
            <td>
                <div class="model-cell">
                    <span class="model-name">{name}</span>
                    <span class="model-provider"><span class="provider-icon"></span> {provider}</span>
                </div>
            </td>
            <td class="elo">{int(item['elo'])}</td>
            {split_elo_cells}
            <td class="win-rate" style="color: {wr_color}">{item['win_rate']:.1f}%</td>
            <td class="matches">{item['matches']}</td>
        </tr>
        """
        rows_html += row

    split_headers = ""
    if show_split_elo:
        split_headers = """
                    <th>Elo-Alt (Wolf/Spy)</th>
                    <th>Elo-Main (Vil/Civ)</th>
        """
    else:
        split_headers = """
                    <th></th>
                    <th></th>
        """

    table_html = f"""
    <div class="leaderboard-section">
        <h2>{title}</h2>
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Model</th>
                    <th>Elo</th>
                    {split_headers}
                    <th>Win Rate</th>
                    <th style="text-align: right;">Matches</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </div>
    """
    return table_html


def generate_cooperative_table_html(title: str, stats: list[dict[str, Any]]) -> str:
    """Generates a win-rate-only table for cooperative/coordination games."""
    rows_html = ""
    # Re-rank by win rate
    sorted_stats = sorted(stats, key=lambda x: x["win_rate"], reverse=True)
    for rank, item in enumerate(sorted_stats, 1):
        rank_display = f"#{rank}"
        if rank == 1:
            rank_display = "🥇"
        if rank == 2:
            rank_display = "🥈"
        if rank == 3:
            rank_display = "🥉"

        name = item["model"]
        provider = "Unknown"
        lower_name = name.lower()
        if "gpt" in lower_name:
            provider = "OpenAI"
        elif "qwen" in lower_name:
            provider = "Alibaba"
        elif "gemini" in lower_name or "google" in lower_name:
            provider = "Google"
        elif "claude" in lower_name:
            provider = "Anthropic"
        elif "llama" in lower_name:
            provider = "Meta"
        elif "mistral" in lower_name:
            provider = "Mistral"

        wr_val = item["win_rate"]
        wr_color = f"hsl({int(wr_val * 1.2)}, 70%, 40%)"

        rows_html += f"""
        <tr>
            <td class="rank">{rank_display}</td>
            <td>
                <div class="model-cell">
                    <span class="model-name">{name}</span>
                    <span class="model-provider"><span class="provider-icon"></span> {provider}</span>
                </div>
            </td>
            <td class="win-rate" style="color: {wr_color}">{wr_val:.1f}%</td>
            <td class="matches">{item['matches']}</td>
        </tr>
        """

    return f"""
    <div class="leaderboard-section">
        <h2>{title}</h2>
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Model</th>
                    <th>Win Rate</th>
                    <th style="text-align: right;">Matches</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </div>
    """


def generate_html_report(
    competitive_tables: Dict[str, list[dict[str, Any]]],
    cooperative_tables: Dict[str, list[dict[str, Any]]],
) -> str:
    """
    Generates the full HTML report.
    competitive_tables / cooperative_tables: { "Title": stats_list, ... }
    """

    def render_competitive_section(
        tables: Dict[str, list[dict[str, Any]]], overall_key: str
    ) -> str:
        html = ""
        if overall_key in tables:
            html += generate_single_table_html(
                f"{overall_key} Leaderboard", tables[overall_key], show_split_elo=True
            )
        for title in sorted(t for t in tables if t != overall_key):
            # Convert title back to game_name (e.g. "Rock Paper Scissors" -> "rock_paper_scissors")
            game_name = title.lower().replace(" ", "_")
            html += generate_single_table_html(
                f"{title} Leaderboard",
                tables[title],
                show_split_elo=is_team_game(game_name),
            )
        return html

    def render_cooperative_section(
        tables: Dict[str, list[dict[str, Any]]], overall_key: str
    ) -> str:
        html = ""
        if overall_key in tables:
            html += generate_cooperative_table_html(
                f"{overall_key} Leaderboard", tables[overall_key]
            )
        for title in sorted(t for t in tables if t != overall_key):
            html += generate_cooperative_table_html(
                f"{title} Leaderboard", tables[title]
            )
        return html

    competitive_html = render_competitive_section(
        competitive_tables, "Competitive Overall"
    )
    cooperative_html = render_cooperative_section(
        cooperative_tables, "Cooperative Overall"
    )

    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Elo Leaderboard</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background-color: #ffffff; color: #333; margin: 0; padding: 40px; }}
            h1 {{ font-size: 28px; font-weight: 700; margin-bottom: 30px; display: flex; align-items: center; gap: 10px; border-bottom: 2px solid #eee; padding-bottom: 20px; }}
            h1::before {{ content: "🏆"; font-size: 32px; }}
            h2 {{ font-size: 20px; font-weight: 600; margin-top: 40px; margin-bottom: 15px; color: #444; }}
            h3 {{ font-size: 16px; font-weight: 600; color: #888; margin: 10px 0 5px; text-transform: uppercase; letter-spacing: 1px; border-left: 4px solid #ccc; padding-left: 10px; }}
            .section-divider {{ margin: 60px 0 30px; border-top: 2px dashed #eee; padding-top: 30px; }}
            table {{ width: 100%; border-collapse: collapse; min-width: 800px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); border: 1px solid #f0f0f0; border-radius: 8px; overflow: hidden; }}
            th {{ text-align: left; font-size: 12px; font-weight: 700; text-transform: uppercase; color: #666; padding: 12px 16px; background-color: #f9f9f9; border-bottom: 1px solid #eee; }}
            td {{ padding: 12px 16px; border-bottom: 1px solid #f5f5f5; vertical-align: middle; }}
            tr:last-child td {{ border-bottom: none; }}
            .rank {{ width: 60px; font-weight: 700; color: #555; font-size: 16px; }}
            .model-cell {{ display: flex; flex-direction: column; }}
            .model-name {{ font-weight: 700; font-size: 15px; color: #000; }}
            .model-provider {{ font-size: 11px; color: #888; display: flex; align-items: center; gap: 4px; margin-top: 2px; }}
            .elo {{ font-weight: 700; font-size: 15px; width: 80px; }}
            .elo-split {{ font-weight: 500; font-size: 14px; width: 100px; color: #666; }}
            .win-rate {{ font-weight: 700; font-size: 15px; width: 100px; }}
            .matches {{ font-weight: 500; font-size: 14px; width: 80px; text-align: right; color: #666; }}
            .footer {{ margin-top: 50px; font-size: 13px; color: #888; border-top: 1px solid #eee; padding-top: 20px; }}
            .provider-icon {{ width: 10px; height: 10px; border-radius: 50%; background-color: #ddd; display: inline-block; }}
            /* Rank Colors */
            tr:nth-child(1) .rank {{ color: #d4af37; }}
            tr:nth-child(2) .rank {{ color: #c0c0c0; }}
            tr:nth-child(3) .rank {{ color: #cd7f32; }}
        </style>
    </head>
    <body>
        <h1>Social Games Tournament Results</h1>

        <h3>⚔️ Competitive Games</h3>
        {competitive_html}

        <div class="section-divider">
            <h3>🤝 Cooperative / Coordination Games</h3>
            <p style="color:#888; font-size:13px; margin-bottom:20px;">
                These games test coordination ability rather than competitive skill.
                Elo here reflects how well a model navigates coordination under conflicting preferences.
            </p>
            {cooperative_html}
        </div>

        <div class="footer">
            <p><strong>Metrics Explanation:</strong></p>
            <ul>
                <li><strong>Elo:</strong> Rating computed from pairwise comparisons within each game category.</li>
                <li><strong>Elo-Alt:</strong> Rating as the minority/hidden role (Werewolf, Spy, Undercover).</li>
                <li><strong>Elo-Main:</strong> Rating as the majority role (Villager, Non-Spy, Civilian).</li>
                <li><strong>Win Rate:</strong> Fraction of pairwise comparisons won (~50% expected for equal competition).</li>
                <li>Cooperative games (Battle of the Sexes, Stag Hunt) are excluded from the competitive Elo.</li>
            </ul>
        </div>
    </body>
    </html>
    """
    return html_template


def process_logs(log_files: list[str]) -> list[dict[str, Any]]:
    """Compute online-ELO ratings from a list of episode log files.

    For each cross-model pair in each episode, applies a K_FACTOR step to the
    overall ELO and (for team games) a separate alt/main split ELO. Same-reward
    pairs are skipped: they carry no head-to-head signal and would artificially
    cap strong models' ratings.

    Runs ELO_EPOCHS shuffled passes over the data. The result is order-dependent
    and noisy when ELO_EPOCHS=1 — use `process_logs_bt` for a stable estimate.
    """
    parsed: list[ParsedEpisode] = []
    for f in log_files:
        ep = _parse_episode(f)
        if ep is not None:
            parsed.append(ep)

    elo_overall: dict[str, float] = defaultdict(lambda: STARTING_ELO)
    elo_alt: dict[str, float] = defaultdict(lambda: STARTING_ELO)
    elo_main: dict[str, float] = defaultdict(lambda: STARTING_ELO)
    wins: dict[str, int] = defaultdict(int)
    total_pairwise: dict[str, int] = defaultdict(int)
    episodes_played: dict[str, int] = defaultdict(int)

    # Count episodes once (independent of epoch shuffling).
    for ep in parsed:
        for m in ep.unique_models:
            episodes_played[m] += 1

    for epoch in range(ELO_EPOCHS):
        shuffled = list(parsed)
        random.shuffle(shuffled)
        first_epoch = epoch == 0

        for ep in shuffled:
            for a1, a2, team_pair in ep.pairs:
                m1 = ep.model_mapping[a1]
                m2 = ep.model_mapping[a2]
                if m1 == m2:
                    continue

                r1 = ep.agent_rewards[a1]
                r2 = ep.agent_rewards[a2]
                if r1 == r2:
                    continue  # ties carry no signal in online ELO

                s1 = 1.0 if r1 > r2 else 0.0
                s2 = 1.0 - s1

                if first_epoch:
                    total_pairwise[m1] += 1
                    total_pairwise[m2] += 1
                    wins[m1 if s1 == 1.0 else m2] += 1

                exp1 = expected_score(elo_overall[m1], elo_overall[m2])
                elo_overall[m1] += K_FACTOR * (s1 - exp1)
                elo_overall[m2] += K_FACTOR * (s2 - (1.0 - exp1))

                if team_pair:
                    exp_alt = expected_score(elo_alt[m1], elo_main[m2])
                    elo_alt[m1] += K_FACTOR * (s1 - exp_alt)
                    elo_main[m2] += K_FACTOR * (s2 - (1.0 - exp_alt))

    models = sorted(episodes_played.keys())
    return _build_stats(
        models, elo_overall, elo_alt, elo_main, wins, total_pairwise, episodes_played
    )


def process_logs_bt(log_files: list[str]) -> list[dict[str, Any]]:
    """Compute Chatbot Arena-style BT ratings (L2-regularized logistic regression).

    Aggregates pairwise win + tie counts across all episodes, then fits
    `compute_mle_elo` once per rating scope (overall, alt-role, main-role).
    Order-independent and stable; the recommended method for offline analysis.
    """
    parsed: list[ParsedEpisode] = []
    for f in log_files:
        ep = _parse_episode(f)
        if ep is not None:
            parsed.append(ep)

    overall_wins: Dict[tuple, int] = defaultdict(int)
    overall_ties: Dict[tuple, int] = defaultdict(int)
    # Role-tagged BT: each player is (model, "alt"|"main"); cross-team battles
    # contribute outcomes between role-tagged identities so wins AND losses
    # on each side are properly fit by a single shared BT model.
    role_wins: Dict[tuple, int] = defaultdict(int)
    role_ties: Dict[tuple, int] = defaultdict(int)

    wins: Dict[str, int] = defaultdict(int)
    total_pairwise: Dict[str, int] = defaultdict(int)
    episodes_played: Dict[str, int] = defaultdict(int)
    all_models: set[str] = set()

    for ep in parsed:
        for m in ep.unique_models:
            episodes_played[m] += 1
            all_models.add(m)

        for a1, a2, team_pair in ep.pairs:
            m1 = ep.model_mapping[a1]
            m2 = ep.model_mapping[a2]
            if m1 == m2:
                continue
            r1 = ep.agent_rewards[a1]
            r2 = ep.agent_rewards[a2]
            total_pairwise[m1] += 1
            total_pairwise[m2] += 1

            if r1 > r2:
                overall_wins[(m1, m2)] += 1
                wins[m1] += 1
                if team_pair:
                    role_wins[((m1, "alt"), (m2, "main"))] += 1
            elif r2 > r1:
                overall_wins[(m2, m1)] += 1
                wins[m2] += 1
                if team_pair:
                    role_wins[((m2, "main"), (m1, "alt"))] += 1
            else:
                # Tie: record under a canonical ordering so we don't double-count.
                a, b = sorted((m1, m2))
                overall_ties[(a, b)] += 1
                if team_pair:
                    pa = (m1, "alt")
                    pb = (m2, "main")
                    ra, rb = sorted((pa, pb))
                    role_ties[(ra, rb)] += 1

    if not all_models:
        return []

    models = sorted(all_models)
    n = len(models)
    overall_elo, ci_lo, ci_hi = bootstrap_mle_elo(overall_wins, overall_ties, models)

    # Single BT fit over (model, role) identities so alt and main ratings
    # share a scale and properly use both wins and losses on each side.
    role_players = sorted({p for k in role_wins for p in k} | {p for k in role_ties for p in k})
    if role_players:
        role_elo_arr = compute_mle_elo(role_wins, role_ties, role_players)
        role_elo = {rp: float(e) for rp, e in zip(role_players, role_elo_arr)}
    else:
        role_elo = {}
    alt_elo_dict: Dict[str, float] = {
        m: role_elo.get((m, "alt"), float(STARTING_ELO)) for m in models
    }
    main_elo_dict: Dict[str, float] = {
        m: role_elo.get((m, "main"), float(STARTING_ELO)) for m in models
    }

    return _build_stats(
        models,
        overall_elo,
        alt_elo_dict,
        main_elo_dict,
        wins,
        total_pairwise,
        episodes_played,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
    )


def save_to_csv(title: str, stats: list[dict[str, Any]], method: str = "bt") -> None:
    """Saves the stats list to a CSV file. method='bt' prefixes the filename
    and adds ELO_lo / ELO_hi columns (95% bootstrap CI) when available."""
    safe_title = title.lower().replace(" ", "_").replace("/", "_")
    prefix = "elo_results_bt_" if method == "bt" else "elo_results_"
    filename = os.path.join("experiments", f"{prefix}{safe_title}.csv")

    has_ci = bool(stats) and "elo_lo" in stats[0]
    headers = ["Rank", "Model", "Elo"]
    if has_ci:
        headers += ["Elo_lo", "Elo_hi"]
    headers += [
        "Elo-Alt (Wolf/Spy)",
        "Elo-Main (Vil/Civ)",
        "Win Rate",
        "Matches",
    ]

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for item in stats:
            row = [item["rank"], item["model"], int(item["elo"])]
            if has_ci:
                row += [int(item["elo_lo"]), int(item["elo_hi"])]
            row += [
                int(item["elo_w"]),
                int(item["elo_v"]),
                f"{item['win_rate']:.1f}%",
                item["matches"],
            ]
            writer.writerow(row)
    print(f"Generated CSV: {filename}")


def calculate_elo(log_dir: str = "logs/tournament", method: str = "bt") -> None:
    """Compute ratings from a directory of episode logs.

    method='bt' (default): Bradley-Terry MLE via L2-regularized logistic
        regression (Chatbot Arena style). Closed-form, order-independent,
        stable; reports 95% bootstrap CIs.
    method='online': online Elo with K_FACTOR step and ELO_EPOCHS shuffled
        passes. Order-dependent; kept for backwards comparison only.
    """
    assert method in ("online", "bt"), f"unknown method: {method}"
    proc = process_logs_bt if method == "bt" else process_logs
    print(f"Calculating ratings ({method}) from logs in: {log_dir}")

    log_files = glob.glob(os.path.join(log_dir, "**", "*.json"), recursive=True)
    print(f"Found {len(log_files)} items")

    # 1. Group logs by game, splitting competitive vs cooperative
    competitive_by_game: Dict[str, list[str]] = defaultdict(list)
    cooperative_by_game: Dict[str, list[str]] = defaultdict(list)

    for filepath in log_files:
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            metadata = data.get("metadata", {})
            game_name = metadata.get("game_name", "Unknown")
            bucket = (
                cooperative_by_game
                if game_name in COOPERATIVE_GAMES
                else competitive_by_game
            )
            bucket[game_name].append(filepath)
        except Exception:
            continue

    competitive_logs = [f for files in competitive_by_game.values() for f in files]
    cooperative_logs = [f for files in cooperative_by_game.values() for f in files]

    # 2. Competitive tables
    competitive_tables: Dict[str, list[dict[str, Any]]] = {}

    print("Processing Competitive Overall...")
    competitive_tables["Competitive Overall"] = proc(competitive_logs)
    save_to_csv(
        "Competitive Overall", competitive_tables["Competitive Overall"], method=method
    )

    for game_name, game_logs in sorted(competitive_by_game.items()):
        if not game_name or game_name == "Unknown":
            continue
        print(f"Processing {game_name} ({len(game_logs)} games)...")
        title = game_name.replace("_", " ").title()
        stats = proc(game_logs)
        competitive_tables[title] = stats
        save_to_csv(title, stats, method=method)

    # 3. Cooperative tables
    cooperative_tables: Dict[str, list[dict[str, Any]]] = {}

    if cooperative_logs:
        print("Processing Cooperative Overall...")
        cooperative_tables["Cooperative Overall"] = proc(cooperative_logs)
        save_to_csv(
            "Cooperative Overall",
            cooperative_tables["Cooperative Overall"],
            method=method,
        )

        for game_name, game_logs in sorted(cooperative_by_game.items()):
            print(f"Processing {game_name} ({len(game_logs)} games)...")
            title = game_name.replace("_", " ").title()
            stats = proc(game_logs)
            cooperative_tables[title] = stats
            save_to_csv(title, stats, method=method)

    # 4. Generate HTML
    html_content = generate_html_report(competitive_tables, cooperative_tables)
    html_name = "elo_leaderboard_bt.html" if method == "bt" else "elo_leaderboard.html"
    output_html = os.path.join("experiments", html_name)
    with open(output_html, "w") as f:
        f.write(html_content)

    print(f"\nSuccessfully generated {output_html}")
    all_titles = list(competitive_tables) + list(cooperative_tables)
    print("Tables generated for:", ", ".join(all_titles))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute model ratings from episode logs."
    )
    parser.add_argument(
        "--log-dir", default="logs/tournament", help="Directory of episode logs."
    )
    parser.add_argument(
        "--method",
        choices=("online", "bt"),
        default="bt",
        help="Rating method: 'bt' = Bradley-Terry MLE (default, recommended), 'online' = online Elo updates.",
    )
    args = parser.parse_args()
    calculate_elo(log_dir=args.log_dir, method=args.method)
