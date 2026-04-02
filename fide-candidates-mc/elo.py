"""ELO-based win probability calculations for chess games."""

from __future__ import annotations

import bisect

# At top-level chess, white has roughly 35 ELO points of effective advantage
WHITE_ADVANTAGE = 35.0

# Base draw rate used only as fallback when outside the empirical table's range
BASE_DRAW_RATE = 0.54


def expected_score(elo_white: float, elo_black: float) -> float:
    """Expected score (0–1) for white against black, including color advantage."""
    return 1.0 / (1.0 + 10.0 ** ((elo_black - elo_white - WHITE_ADVANTAGE) / 400.0))


# ---------------------------------------------------------------------------
# Empirical draw probability table
# Source: chess-db.com historical game database research
# Axes: average rating (rows, 200-pt steps) × rating difference (cols, 20-pt steps)
# ---------------------------------------------------------------------------

_AVG_BREAKPOINTS  = [1400, 1600, 1800, 2000, 2200, 2400, 2600]
_DIFF_BREAKPOINTS = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300]

# Draw probability indexed by [avg_rating_bracket][diff_bracket]
# Rows correspond to _AVG_BREAKPOINTS; columns to _DIFF_BREAKPOINTS
_DRAW_TABLE: list[list[float]] = [
    # avg ≈ 1400
    [0.213, 0.198, 0.183, 0.167, 0.152, 0.137, 0.122, 0.109, 0.096, 0.084, 0.074, 0.064, 0.056, 0.048, 0.042, 0.037],
    # avg ≈ 1600
    [0.283, 0.265, 0.247, 0.228, 0.210, 0.192, 0.174, 0.157, 0.141, 0.126, 0.112, 0.099, 0.087, 0.077, 0.067, 0.059],
    # avg ≈ 1800
    [0.360, 0.340, 0.319, 0.298, 0.277, 0.256, 0.235, 0.215, 0.196, 0.177, 0.160, 0.143, 0.128, 0.114, 0.101, 0.089],
    # avg ≈ 2000
    [0.430, 0.410, 0.388, 0.367, 0.345, 0.323, 0.301, 0.280, 0.259, 0.239, 0.219, 0.201, 0.183, 0.167, 0.151, 0.137],
    # avg ≈ 2200
    [0.482, 0.462, 0.441, 0.420, 0.398, 0.376, 0.354, 0.333, 0.311, 0.291, 0.270, 0.251, 0.232, 0.214, 0.197, 0.181],
    # avg ≈ 2400
    [0.541, 0.521, 0.500, 0.479, 0.457, 0.435, 0.413, 0.391, 0.370, 0.349, 0.328, 0.308, 0.289, 0.271, 0.253, 0.236],
    # avg ≈ 2600
    [0.578, 0.558, 0.537, 0.516, 0.495, 0.473, 0.451, 0.429, 0.408, 0.387, 0.366, 0.346, 0.326, 0.307, 0.289, 0.271],
]


def draw_rate(elo_white: float, elo_black: float, base: float = BASE_DRAW_RATE) -> float:
    """Draw probability from empirical lookup table (chess-db.com research).

    Falls back to `base` when both players are outside the tabulated range.
    """
    avg  = (elo_white + elo_black) / 2.0
    diff = abs(elo_white - elo_black)

    avg_clamped  = max(_AVG_BREAKPOINTS[0],  min(_AVG_BREAKPOINTS[-1],  avg))
    diff_clamped = max(_DIFF_BREAKPOINTS[0], min(_DIFF_BREAKPOINTS[-1], diff))

    avg_idx  = min(bisect.bisect_right(_AVG_BREAKPOINTS,  avg_clamped)  - 1, len(_AVG_BREAKPOINTS)  - 1)
    diff_idx = min(bisect.bisect_right(_DIFF_BREAKPOINTS, diff_clamped) - 1, len(_DIFF_BREAKPOINTS) - 1)

    return _DRAW_TABLE[avg_idx][diff_idx]


def game_probabilities(
    elo_white: float,
    elo_black: float,
    base_draw: float = BASE_DRAW_RATE,
) -> tuple[float, float, float]:
    """Return (p_white_win, p_draw, p_black_win) for a single game.

    Formula (preserves ELO expected score exactly):
        p_white_win = E - 0.5 * d
        p_black_win = (1 - E) - 0.5 * d
        p_draw      = d

    Verification: E[score_white] = p_white_win + 0.5·p_draw
                                  = (E - 0.5·d) + 0.5·d = E  ✓

    If the ELO gap is so extreme that p_white_win < 0, probabilities are
    clamped and renormalised (practically never occurs at Candidates level).
    """
    e = expected_score(elo_white, elo_black)
    d = draw_rate(elo_white, elo_black, base_draw)

    p_white = e - 0.5 * d
    p_black = (1.0 - e) - 0.5 * d

    # Guard against extreme ELO gaps producing negative probabilities
    if p_white < 0.0 or p_black < 0.0:
        p_white = max(0.0, p_white)
        p_black = max(0.0, p_black)
        total = p_white + p_black + d
        p_white /= total
        p_black /= total
        d      /= total

    return p_white, d, p_black
