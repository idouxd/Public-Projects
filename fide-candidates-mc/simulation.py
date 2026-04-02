"""Vectorized Monte Carlo simulation engine.

All n_sims simulations are run in parallel with NumPy, making even
n_sims=500_000 complete in a few seconds.

Tiebreak order (simplified):
  1. Total score
  2. Number of decisive wins
  3. Tiny ELO-proportional random (approximates rapid/blitz playoff odds)
"""

from __future__ import annotations

import numpy as np

from elo import game_probabilities
from tournament import Tournament


def simulate(
    tournament: Tournament,
    n_sims: int = 200_000,
    base_draw: float = 0.58,
    seed: int | None = None,
) -> dict[str, dict]:
    """Run Monte Carlo and return per-player statistics.

    Returns a dict keyed by player_id with:
      - "win_prob":    probability of winning the tournament
      - "median_pts":  median final score across all simulations
      - "q10_pts":     10th-percentile final score
      - "q90_pts":     90th-percentile final score
    """
    player_ids = tournament.player_ids
    players = tournament.players
    n_players = len(player_ids)
    pid_idx = {pid: i for i, pid in enumerate(player_ids)}

    # Current state ---------------------------------------------------------
    standings = tournament.standings()
    base_scores = np.array([standings[pid] for pid in player_ids], dtype=np.float64)

    current_wins = tournament.wins()
    base_wins = np.array([current_wins[pid] for pid in player_ids], dtype=np.float64)

    remaining = tournament.remaining_games()

    # Edge case: tournament already complete
    if not remaining:
        max_pts = base_scores.max()
        leaders = [pid for pid in player_ids if standings[pid] == max_pts]
        winner = max(leaders, key=lambda p: (current_wins[p], players[p].elo))
        return {
            pid: {
                "win_prob": 1.0 if pid == winner else 0.0,
                "median_pts": base_scores[pid_idx[pid]],
                "q10_pts": base_scores[pid_idx[pid]],
                "q90_pts": base_scores[pid_idx[pid]],
            }
            for pid in player_ids
        }

    n_games = len(remaining)

    # Precompute probabilities for each remaining game ----------------------
    w_idx = np.array([pid_idx[w] for w, _ in remaining], dtype=np.intp)
    b_idx = np.array([pid_idx[b] for _, b in remaining], dtype=np.intp)

    probs = np.array(
        [game_probabilities(players[w].elo, players[b].elo, base_draw)
         for w, b in remaining]
    )  # shape (n_games, 3): [p_white_win, p_draw, p_black_win]

    p_w = probs[:, 0]                       # (n_games,)
    p_wd = p_w + probs[:, 1]               # cumulative: white_win or draw

    # Simulate all games for all sims at once --------------------------------
    rng = np.random.default_rng(seed)
    rand = rng.random((n_games, n_sims))    # (n_games, n_sims)

    white_delta = np.where(
        rand < p_w[:, None], 1.0,
        np.where(rand < p_wd[:, None], 0.5, 0.0),
    )   # (n_games, n_sims)  — white's score contribution per game
    black_delta = 1.0 - white_delta

    # Accumulate per-player scores across all simulations -------------------
    sim_scores = np.tile(base_scores[:, None], (1, n_sims))  # (n_players, n_sims)
    sim_wins   = np.tile(base_wins[:, None],   (1, n_sims))

    white_win_mask = (rand < p_w[:, None]).astype(np.float64)   # 1 if white wins
    black_win_mask = (rand >= p_wd[:, None]).astype(np.float64) # 1 if black wins

    for p in range(n_players):
        w_mask = (w_idx == p)
        b_mask = (b_idx == p)
        if w_mask.any():
            sim_scores[p] += white_delta[w_mask].sum(axis=0)
            sim_wins[p]   += white_win_mask[w_mask].sum(axis=0)
        if b_mask.any():
            sim_scores[p] += black_delta[b_mask].sum(axis=0)
            sim_wins[p]   += black_win_mask[b_mask].sum(axis=0)

    # Determine tournament winner in each simulation ------------------------
    # Composite ranking: score (primary) → wins (secondary) → ELO noise (tertiary)
    elo_arr = np.array([players[pid].elo for pid in player_ids], dtype=np.float64)
    elo_noise = rng.random((n_players, n_sims)) * 1e-4 * (elo_arr[:, None] / 2800.0)

    composite = sim_scores * 1000.0 + sim_wins + elo_noise
    winner_idx = composite.argmax(axis=0)   # (n_sims,)

    # Win probabilities
    counts = np.bincount(winner_idx, minlength=n_players)

    # Score distribution stats
    results: dict[str, dict] = {}
    for i, pid in enumerate(player_ids):
        pts = sim_scores[i]
        results[pid] = {
            "win_prob":  float(counts[i]) / n_sims,
            "median_pts": float(np.median(pts)),
            "q10_pts":    float(np.percentile(pts, 10)),
            "q90_pts":    float(np.percentile(pts, 90)),
        }

    return results
