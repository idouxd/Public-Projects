"""ELO-based win probability calculations for chess games."""

from __future__ import annotations

# At top-level chess, white has roughly 35 ELO points of effective advantage
WHITE_ADVANTAGE = 35.0

# At equal ELO, roughly 58% of top-level games end in draws
BASE_DRAW_RATE = 0.58


def expected_score(elo_white: float, elo_black: float) -> float:
    """Expected score (0–1) for white against black, including color advantage."""
    return 1.0 / (1.0 + 10.0 ** ((elo_black - elo_white - WHITE_ADVANTAGE) / 400.0))


def draw_rate(elo_white: float, elo_black: float, base: float = BASE_DRAW_RATE) -> float:
    """Draw probability, declining as ELO gap widens.

    Model: draw_rate = max(0.35, base - 0.10 * |elo_diff| / 100)
    Examples (base=0.58):
      diff=   0 → 58%     diff=100 → 48%
      diff= 200 → 38%     diff=250+ → 35% (floor)
    """
    diff = abs(elo_white - elo_black)
    return max(0.35, base - 0.10 * diff / 100.0)


def game_probabilities(
    elo_white: float,
    elo_black: float,
    base_draw: float = BASE_DRAW_RATE,
) -> tuple[float, float, float]:
    """Return (p_white_win, p_draw, p_black_win) for a single game.

    The model satisfies:
        E[score_white] = p_white_win + 0.5 * p_draw = expected_score(...)
    """
    e = expected_score(elo_white, elo_black)
    d = draw_rate(elo_white, elo_black, base_draw)
    p_white = (1.0 - d) * e
    p_black = (1.0 - d) * (1.0 - e)
    return p_white, d, p_black
