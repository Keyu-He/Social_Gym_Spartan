import csv
import functools
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

from experiments.utils import load_game_config, rated_logs, unrecovered_request_failures

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
    # Gemma 3 12B / 4B, Qwen 3 1.7B
    "google/gemma-3-12b-it": "Gemma3-12B",
    "google/gemma-3-4b-it": "Gemma3-4B",
    "Qwen/Qwen3-1.7B": "Qwen3-1.7B",
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


EpisodeCounts = tuple[Dict[tuple, int], Dict[tuple, int]]  # one episode's (wins, ties)

# Resampling unit of the bootstrap behind every 95% interval (bootstrap_mle_elo): "pair" draws battles, the
# leaderboard's default; "episode" draws whole episodes.
BOOTSTRAP_UNITS = ("pair", "episode")
DEFAULT_BOOTSTRAP = "pair"


def _pair_draws(
    win_counts: Dict[tuple, int], tie_counts: Dict[tuple, int], n_bootstrap: int, seed: int
) -> Iterable[EpisodeCounts]:
    """FastChat's compute_bootstrap_bt resample. Each draw takes N battles with replacement, N being the number
    of battles in win_counts and tie_counts together, as one multinomial over all their cells: a tied pair is
    one battle, drawn like a decided one. Cells are taken in sorted order, so the draws do not depend on the
    order in which the counts were built."""
    win_keys, tie_keys = sorted(win_counts), sorted(tie_counts)
    counts = np.array([win_counts[k] for k in win_keys] + [tie_counts[k] for k in tie_keys], dtype=float)
    probs = counts / counts.sum()
    total = int(counts.sum())
    rng = np.random.default_rng(seed)
    for _ in range(n_bootstrap):
        draw = rng.multinomial(total, probs)
        yield (
            {k: int(v) for k, v in zip(win_keys, draw[: len(win_keys)]) if v > 0},
            {k: int(v) for k, v in zip(tie_keys, draw[len(win_keys) :]) if v > 0},
        )


class _EpisodeCountTable:
    """Per-episode win and tie counts as matrices, so a resample of episodes aggregates in one product."""

    def __init__(self, episodes: list[EpisodeCounts]) -> None:
        self.win_keys = sorted({k for w, _ in episodes for k in w})
        self.tie_keys = sorted({k for _, t in episodes for k in t})
        wi = {k: i for i, k in enumerate(self.win_keys)}
        ti = {k: i for i, k in enumerate(self.tie_keys)}
        self.wins = np.zeros((len(episodes), len(self.win_keys)))
        self.ties = np.zeros((len(episodes), len(self.tie_keys)))
        for e, (w, t) in enumerate(episodes):
            for k, c in w.items():
                self.wins[e, wi[k]] += c
            for k, c in t.items():
                self.ties[e, ti[k]] += c

    def aggregate(self, multiplicity: np.ndarray) -> EpisodeCounts:
        """Counts of a resample in which episode e appears multiplicity[e] times."""
        w = multiplicity @ self.wins if self.win_keys else np.zeros(0)
        t = multiplicity @ self.ties if self.tie_keys else np.zeros(0)
        return (
            {k: int(round(v)) for k, v in zip(self.win_keys, w) if v > 0},
            {k: int(round(v)) for k, v in zip(self.tie_keys, t) if v > 0},
        )


def _episode_multiplicities(n_episodes: int, n_bootstrap: int, seed: int) -> Iterable[np.ndarray]:
    """For each bootstrap draw, how often each episode appears in a resample of n_episodes episodes
    drawn with replacement. The same seed gives the same draws, so fits that share a seed (the overall
    and the alt/main fits in process_logs_bt) see the same resampled episodes."""
    rng = np.random.default_rng(seed)
    for _ in range(n_bootstrap):
        yield np.bincount(rng.integers(0, n_episodes, n_episodes), minlength=n_episodes).astype(float)


def bootstrap_mle_elo(
    win_counts: Dict[tuple, int],
    tie_counts: Dict[tuple, int],
    models: list[str],
    n_bootstrap: int = 1000,
    seed: int = 42,
    bootstrap: str = DEFAULT_BOOTSTRAP,
    episodes: Optional[list[EpisodeCounts]] = None,
    **mle_kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bootstrap 95% CI for the BT MLE. Returns (point, ci_lo, ci_hi) aligned with `models`.

    Each of the `n_bootstrap` draws resamples the battles and refits `compute_mle_elo`; the interval is the
    2.5th to 97.5th percentile of the refits. A tie counts as one half-game in every refit, as in the point fit.

    bootstrap="pair" (default) is FastChat's compute_bootstrap_bt: a multinomial draw of as many battles as the
    data has over the win and tie cells together (`_pair_draws`).
    bootstrap="episode" resamples whole episodes with replacement, so all the pairs of one episode are drawn
    together. It needs `episodes`, one (wins, ties) pair per episode summing to win_counts / tie_counts
    (`_episode_pair_counts` builds them); the pair mode does not use them. Fits that share a seed draw the
    same episodes.
    """
    if bootstrap not in BOOTSTRAP_UNITS:
        raise ValueError(f"unknown bootstrap unit {bootstrap!r}; expected one of {BOOTSTRAP_UNITS}")
    if bootstrap == "episode" and episodes is None:
        raise TypeError(
            "the episode bootstrap needs per-episode counts (episodes=[(wins, ties), ...])"
        )
    point = compute_mle_elo(win_counts, tie_counts, models, **mle_kwargs)
    n_battles = sum(win_counts.values()) + sum(tie_counts.values())
    if n_battles == 0 or (bootstrap == "episode" and not episodes):
        return point, point.copy(), point.copy()

    if bootstrap == "pair":
        draws = _pair_draws(win_counts, tie_counts, n_bootstrap, seed)
    else:
        table = _EpisodeCountTable(episodes)
        draws = (table.aggregate(m) for m in _episode_multiplicities(len(episodes), n_bootstrap, seed))
    samples = np.zeros((n_bootstrap, len(models)))
    for b, (wc, tc) in enumerate(draws):
        try:
            samples[b] = compute_mle_elo(wc, tc, models, **mle_kwargs)
        except Exception:
            samples[b] = point  # fallback if a draw produces a degenerate fit
    return point, np.percentile(samples, 2.5, axis=0), np.percentile(samples, 97.5, axis=0)


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
      - a tie is one half-game each way: it adds an (a-wins=1) row and an
        (a-wins=0) row with weight 0.5 each, so a tie weighs as much as one
        decisive result (weight 1) and counts as half a win for each side.
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
        # A tie is half a win for each side: one a-wins row and one b-wins row, half weight each.
        rows.append((idx[a], idx[b], 1.0, 0.5 * float(c)))
        rows.append((idx[a], idx[b], 0.0, 0.5 * float(c)))

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


def _parse_episode(
    filepath: str, skipped: Optional[list[str]] = None
) -> Optional[ParsedEpisode]:
    """Load an episode log and return its data in a form both rating methods
    can iterate over. Returns None for malformed logs (caller should skip).

    Also returns None for an episode in which a model call failed at the
    request level and was not recovered by the retries (a connection error,
    timeout or server error played as a 'none' move): its outcome is not the
    models' play. Such files are appended to `skipped` when it is given; the
    tournament runner replays their rosters on resume. A model's own
    malformed reply does not cause a skip.
    """
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
    except Exception:
        return None

    if unrecovered_request_failures(data):
        if skipped is not None:
            skipped.append(filepath)
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
    wins: Dict[str, float],
    total_pairwise: Dict[str, int],
    episodes_played: Dict[str, int],
    ci_lo: Optional[np.ndarray] = None,
    ci_hi: Optional[np.ndarray] = None,
    role_ci: Optional[Dict[str, Dict[str, float]]] = None,
) -> list[dict[str, Any]]:
    """Build the per-model stats list shared by both rating methods.

    `models` defines the order used to interpret ndarray-typed ratings.
    Output is sorted by descending overall ELO. win_rate is wins / pairwise comparisons, where a tie
    counts as half a win for each side (the BT fit's convention). If `ci_lo`/`ci_hi` are
    provided (BT bootstrap), they're carried through as elo_lo / elo_hi.
    `role_ci` (model -> {"elo_w_lo", "elo_w_hi", "elo_v_lo", "elo_v_hi"}) adds
    the alt/main intervals from the same bootstrap; they stay in the stats
    rows and are not written to the CSV.
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
        if role_ci is not None and m in role_ci:
            row.update(role_ci[m])
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
                    <th>Win Rate (tie = ½)</th>
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
                    <th>Win Rate (tie = ½)</th>
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
                <li><strong>Win Rate (tie = ½):</strong> Fraction of pairwise comparisons won, a tie counting as half a win for each side, as in the Elo fit (~50% expected for equal competition).</li>
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
    parsed = _load_rated_episodes(log_files)

    elo_overall: dict[str, float] = defaultdict(lambda: STARTING_ELO)
    elo_alt: dict[str, float] = defaultdict(lambda: STARTING_ELO)
    elo_main: dict[str, float] = defaultdict(lambda: STARTING_ELO)
    wins: dict[str, float] = defaultdict(float)
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
                    if first_epoch:  # Win Rate counts a tie as half a win for each side
                        total_pairwise[m1] += 1
                        total_pairwise[m2] += 1
                        wins[m1] += 0.5
                        wins[m2] += 0.5
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


def _load_rated_episodes(log_files: list[str]) -> list[ParsedEpisode]:
    """Parse the logs, leaving out malformed ones, episodes with an unrecovered request failure, and all but
    the latest clean log of a roster that has several (experiments.utils.rated_logs); the counts are printed."""
    parsed: list[ParsedEpisode] = []
    skipped: list[str] = []
    duplicates: list[str] = []
    for f in rated_logs(log_files, failed=skipped, duplicates=duplicates):
        ep = _parse_episode(f, skipped=skipped)
        if ep is not None:
            parsed.append(ep)
    if skipped:
        print(
            f"  skipped {len(skipped)} episode(s) with an unrecovered request failure "
            f"(of {len(log_files)} logs)"
        )
    if duplicates:
        print(f"  left out {len(duplicates)} older clean log(s) of a roster rated from its latest log")
    return parsed


def _episode_pair_counts(ep: ParsedEpisode) -> tuple[EpisodeCounts, EpisodeCounts]:
    """One episode's cross-model outcomes: (wins, ties) by model and, for cross-team pairs of a team
    game, (wins, ties) by (model, "alt"|"main"). Same-model pairs are skipped; ties are keyed under a
    canonical (sorted) order."""
    wins: Dict[tuple, int] = defaultdict(int)
    ties: Dict[tuple, int] = defaultdict(int)
    role_wins: Dict[tuple, int] = defaultdict(int)
    role_ties: Dict[tuple, int] = defaultdict(int)
    for a1, a2, team_pair in ep.pairs:
        m1 = ep.model_mapping[a1]
        m2 = ep.model_mapping[a2]
        if m1 == m2:
            continue
        r1 = ep.agent_rewards[a1]
        r2 = ep.agent_rewards[a2]
        if r1 > r2:
            wins[(m1, m2)] += 1
            if team_pair:
                role_wins[((m1, "alt"), (m2, "main"))] += 1
        elif r2 > r1:
            wins[(m2, m1)] += 1
            if team_pair:
                role_wins[((m2, "main"), (m1, "alt"))] += 1
        else:
            ties[tuple(sorted((m1, m2)))] += 1
            if team_pair:
                role_ties[tuple(sorted(((m1, "alt"), (m2, "main"))))] += 1
    return (dict(wins), dict(ties)), (dict(role_wins), dict(role_ties))


def process_logs_bt(
    log_files: list[str],
    n_bootstrap: int = 1000,
    seed: int = 42,
    bootstrap: str = DEFAULT_BOOTSTRAP,
) -> list[dict[str, Any]]:
    """Compute Chatbot Arena-style BT ratings (L2-regularized logistic regression).

    Aggregates pairwise win + tie counts across all episodes, then fits
    `compute_mle_elo` once per rating scope (overall, and one joint fit over
    (model, alt|main) identities for the role ratings). A tie counts as half a
    win for each side. The 95% intervals come from `n_bootstrap` bootstrap
    draws (`bootstrap_mle_elo`). By default (bootstrap="pair") each draw is a
    multinomial resample of the battles over the win and tie cells, as in
    FastChat; bootstrap="episode" resamples whole episodes instead, the same
    episodes for the overall and the role fit. Episodes with an unrecovered
    request failure are left out. Order-independent and stable; the
    recommended method for offline analysis.
    """
    parsed = _load_rated_episodes(log_files)

    overall_wins: Dict[tuple, int] = defaultdict(int)
    overall_ties: Dict[tuple, int] = defaultdict(int)
    # Role-tagged BT: each player is (model, "alt"|"main"); cross-team battles
    # contribute outcomes between role-tagged identities so wins AND losses
    # on each side are properly fit by a single shared BT model.
    role_wins: Dict[tuple, int] = defaultdict(int)
    role_ties: Dict[tuple, int] = defaultdict(int)
    per_episode: list[EpisodeCounts] = []
    per_episode_role: list[EpisodeCounts] = []

    wins: Dict[str, float] = defaultdict(float)  # Win Rate numerator: a tie is half a win for each side
    total_pairwise: Dict[str, int] = defaultdict(int)
    episodes_played: Dict[str, int] = defaultdict(int)
    all_models: set[str] = set()

    for ep in parsed:
        for m in ep.unique_models:
            episodes_played[m] += 1
            all_models.add(m)

        (ew, et), (rw, rt) = _episode_pair_counts(ep)
        per_episode.append((ew, et))
        per_episode_role.append((rw, rt))
        for (w, l), c in ew.items():
            overall_wins[(w, l)] += c
            wins[w] += c
            total_pairwise[w] += c
            total_pairwise[l] += c
        for (a, b), c in et.items():
            overall_ties[(a, b)] += c
            wins[a] += 0.5 * c
            wins[b] += 0.5 * c
            total_pairwise[a] += c
            total_pairwise[b] += c
        for k, c in rw.items():
            role_wins[k] += c
        for k, c in rt.items():
            role_ties[k] += c

    if not all_models:
        return []

    models = sorted(all_models)
    overall_elo, ci_lo, ci_hi = bootstrap_mle_elo(
        overall_wins,
        overall_ties,
        models,
        n_bootstrap=n_bootstrap,
        seed=seed,
        bootstrap=bootstrap,
        episodes=per_episode,
    )

    # Single BT fit over (model, role) identities so alt and main ratings
    # share a scale and properly use both wins and losses on each side. Its
    # intervals use the same bootstrap unit and seed as the overall ones
    # above (with the episode bootstrap, the same resampled episodes).
    role_players = sorted(
        {p for k in role_wins for p in k} | {p for k in role_ties for p in k}
    )
    role_ci: Dict[str, Dict[str, float]] = {}
    if role_players:
        role_elo_arr, role_lo, role_hi = bootstrap_mle_elo(
            role_wins,
            role_ties,
            role_players,
            n_bootstrap=n_bootstrap,
            seed=seed,
            bootstrap=bootstrap,
            episodes=per_episode_role,
        )
        role_elo = {rp: float(e) for rp, e in zip(role_players, role_elo_arr)}
        for rp, lo, hi in zip(role_players, role_lo, role_hi):
            side = "w" if rp[1] == "alt" else "v"
            role_ci.setdefault(rp[0], {})[f"elo_{side}_lo"] = float(lo)
            role_ci[rp[0]][f"elo_{side}_hi"] = float(hi)
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
        role_ci=role_ci or None,
    )


def save_to_csv(
    title: str, stats: list[dict[str, Any]], method: str = "bt", out_dir: str = "experiments"
) -> None:
    """Saves the stats list to a CSV file. method='bt' prefixes the filename
    and adds ELO_lo / ELO_hi columns (95% bootstrap CI) when available."""
    safe_title = title.lower().replace(" ", "_").replace("/", "_")
    prefix = "elo_results_bt_" if method == "bt" else "elo_results_"
    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.join(out_dir, f"{prefix}{safe_title}.csv")

    has_ci = bool(stats) and "elo_lo" in stats[0]
    # "Win Rate" counts a tie as half a win for each side (the HTML header says so).
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


def calculate_elo(
    log_dir: str = "logs/tournament",
    method: str = "bt",
    out_dir: str = "experiments",
    bootstrap: str = DEFAULT_BOOTSTRAP,
) -> None:
    """Compute ratings from a directory of episode logs.

    method='bt' (default): Bradley-Terry MLE via L2-regularized logistic
        regression (Chatbot Arena style). Closed-form, order-independent,
        stable; reports 95% bootstrap CIs. bootstrap='pair' (default)
        resamples battles over the win and tie cells, as FastChat does;
        bootstrap='episode' resamples whole episodes.
    method='online': online Elo with K_FACTOR step and ELO_EPOCHS shuffled
        passes. Order-dependent; kept for backwards comparison only.

    Episodes with an unrecovered request-level failure are skipped and
    counted, and a roster with several clean logs is rated from its latest
    one (experiments.utils.rated_logs). A log of a rule variant
    (metadata.variant, e.g. the no-discussion communication control) is
    never rated. CSVs and the HTML leaderboard are written to `out_dir`.
    """
    assert method in ("online", "bt"), f"unknown method: {method}"
    assert bootstrap in BOOTSTRAP_UNITS, f"unknown bootstrap unit: {bootstrap}"
    if method == "bt":
        proc = functools.partial(process_logs_bt, bootstrap=bootstrap)
        print(f"Calculating ratings (bt, {bootstrap} bootstrap) from logs in: {log_dir}")
    else:
        proc = process_logs
        print(f"Calculating ratings ({method}) from logs in: {log_dir}")

    log_files = glob.glob(os.path.join(log_dir, "**", "*.json"), recursive=True)
    print(f"Found {len(log_files)} items")

    # 1. Group logs by game, splitting competitive vs cooperative
    competitive_by_game: Dict[str, list[str]] = defaultdict(list)
    cooperative_by_game: Dict[str, list[str]] = defaultdict(list)
    failed_by_game: Dict[str, int] = defaultdict(int)
    variant_logs: Dict[str, int] = defaultdict(int)
    no_game: list[str] = []

    for filepath in log_files:
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            metadata = data.get("metadata", {})
            game_name = metadata.get("game_name", "Unknown")
            if metadata.get("variant"):
                # A rule variant is another game; left out before the one-log-per-roster selection, so a
                # variant log of a leaderboard roster can never stand in for the roster's own log.
                variant_logs[str(metadata["variant"])] += 1
                continue
            if unrecovered_request_failures(data):
                # A model call failed at the request level and was played as 'none'; the
                # runner replays the roster on resume, so the episode is not rated.
                failed_by_game[game_name] += 1
                continue
            if not metadata.get("game_name"):
                # No per-game table can show it, so it is left out of the overall tables too.
                no_game.append(filepath)
                continue
            bucket = (
                cooperative_by_game
                if game_name in COOPERATIVE_GAMES
                else competitive_by_game
            )
            bucket[game_name].append(filepath)
        except Exception:
            continue
    if no_game:
        print(f"Left out {len(no_game)} log(s) without metadata.game_name (no table can show them)")
    if variant_logs:
        print(
            f"Left out {sum(variant_logs.values())} log(s) of a rule variant ("
            + ", ".join(f"{v} {n}" for v, n in sorted(variant_logs.items()))
            + ")"
        )

    n_failed = sum(failed_by_game.values())
    print(
        f"Skipped {n_failed} episode(s) with an unrecovered request failure"
        + (
            ": " + ", ".join(f"{g} {n}" for g, n in sorted(failed_by_game.items()))
            if n_failed
            else ""
        )
    )
    duplicates: list[str] = []
    rated_logs([f for files in competitive_by_game.values() for f in files]
               + [f for files in cooperative_by_game.values() for f in files], duplicates=duplicates)
    print(
        f"Left out {len(duplicates)} older clean log(s) of a roster that has a later clean log "
        f"(one log is rated per roster)"
    )

    competitive_logs = [f for files in competitive_by_game.values() for f in files]
    cooperative_logs = [f for files in cooperative_by_game.values() for f in files]

    # 2. Competitive tables
    competitive_tables: Dict[str, list[dict[str, Any]]] = {}

    print("Processing Competitive Overall...")
    competitive_tables["Competitive Overall"] = proc(competitive_logs)
    save_to_csv(
        "Competitive Overall",
        competitive_tables["Competitive Overall"],
        method=method,
        out_dir=out_dir,
    )

    for game_name, game_logs in sorted(competitive_by_game.items()):
        if not game_name or game_name == "Unknown":
            continue
        print(f"Processing {game_name} ({len(game_logs)} games)...")
        title = game_name.replace("_", " ").title()
        stats = proc(game_logs)
        competitive_tables[title] = stats
        save_to_csv(title, stats, method=method, out_dir=out_dir)

    # 3. Cooperative tables
    cooperative_tables: Dict[str, list[dict[str, Any]]] = {}

    if cooperative_logs:
        print("Processing Cooperative Overall...")
        cooperative_tables["Cooperative Overall"] = proc(cooperative_logs)
        save_to_csv(
            "Cooperative Overall",
            cooperative_tables["Cooperative Overall"],
            method=method,
            out_dir=out_dir,
        )

        for game_name, game_logs in sorted(cooperative_by_game.items()):
            print(f"Processing {game_name} ({len(game_logs)} games)...")
            title = game_name.replace("_", " ").title()
            stats = proc(game_logs)
            cooperative_tables[title] = stats
            save_to_csv(title, stats, method=method, out_dir=out_dir)

    # 4. Generate HTML
    html_content = generate_html_report(competitive_tables, cooperative_tables)
    html_name = "elo_leaderboard_bt.html" if method == "bt" else "elo_leaderboard.html"
    output_html = os.path.join(out_dir, html_name)
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
    parser.add_argument(
        "--out-dir",
        default="experiments",
        help="Directory for the CSVs and the HTML leaderboard (default: experiments).",
    )
    parser.add_argument(
        "--bootstrap",
        choices=BOOTSTRAP_UNITS,
        default=DEFAULT_BOOTSTRAP,
        help="Resampling unit of the 95%% CIs (bt only): 'pair' = battles, a multinomial over the win and tie "
        "cells as in FastChat (default); 'episode' = whole episodes.",
    )
    args = parser.parse_args()
    calculate_elo(
        log_dir=args.log_dir, method=args.method, out_dir=args.out_dir, bootstrap=args.bootstrap
    )
