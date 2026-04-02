"""Plot win probability for each player after every completed round.

Usage
-----
  python plot_probabilities.py open            # 2026 Open section
  python plot_probabilities.py womens          # Women's section
  python plot_probabilities.py open --sims 200000
  python plot_probabilities.py open --save     # save PNG instead of showing
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from config import SECTIONS
from tournament import Tournament, GameResult
from simulation import simulate

DATA_DIR = Path(__file__).parent / "data"

# Colour palette — one per player (up to 8)
PALETTE = [
    "#E63946",  # red
    "#457B9D",  # steel blue
    "#2A9D8F",  # teal
    "#E9C46A",  # amber
    "#F4A261",  # orange
    "#6A4C93",  # purple
    "#264653",  # dark teal
    "#A8DADC",  # light blue
]


def snapshot_tournament(base: Tournament, max_round: int) -> Tournament:
    """Return a tournament whose results are filtered to rounds <= max_round."""
    t = Tournament.__new__(Tournament)
    t.players = base.players
    t.player_ids = base.player_ids
    t.results_path = base.results_path
    t.results = [r for r in base.results if r.round <= max_round]
    return t


def collect_probabilities(
    section: str,
    n_sims: int,
) -> tuple[list[int], dict[str, list[float]]]:
    """Simulate after every round (including round 0 = pre-tournament).

    Returns
    -------
    rounds : list of round numbers [0, 1, 2, …]
    probs  : dict player_id -> list of win probabilities, one per round
    """
    players = SECTIONS[section]
    results_path = DATA_DIR / f"{section}_results.json"
    full = Tournament(players, results_path)

    all_rounds = sorted({r.round for r in full.results if r.round > 0})
    round_labels = [0] + all_rounds   # 0 = no games played

    probs: dict[str, list[float]] = {pid: [] for pid in full.player_ids}

    for rnd in round_labels:
        snap = snapshot_tournament(full, rnd)
        results = simulate(snap, n_sims=n_sims)
        for pid in full.player_ids:
            probs[pid].append(results[pid]["win_prob"] * 100)
        played = snap.games_played()
        print(f"  Round {rnd:>2}  ({played:>2} games)  simulated.")

    return round_labels, probs


def make_plot(
    section: str,
    round_labels: list[int],
    probs: dict[str, list[float]],
    players_list,
    save: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 6.5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    n_players = len(players_list)
    colors = PALETTE[:n_players]

    # Sort players by final probability (descending) for legend order
    sorted_players = sorted(
        players_list,
        key=lambda p: probs[p.id][-1],
        reverse=True,
    )

    x = round_labels

    for i, player in enumerate(sorted_players):
        color = colors[i % len(colors)]
        y = probs[player.id]
        final = y[-1]

        ax.plot(
            x, y,
            color=color,
            linewidth=2.2,
            marker="o",
            markersize=5,
            markerfacecolor=color,
            markeredgecolor="#0d1117",
            markeredgewidth=1.2,
            zorder=3,
        )

        # Label at the right end
        ax.annotate(
            f"{player.name.split()[-1]}  {final:.1f}%",
            xy=(x[-1], y[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            color=color,
            fontsize=8.5,
            fontweight="bold",
        )

    # Dashed 50% guide line
    ax.axhline(50, color="#ffffff22", linewidth=0.8, linestyle="--")

    # Axes styling
    title_section = "Open" if section == "open" else "Women's"
    ax.set_title(
        f"2026 FIDE Candidates ({title_section}) — Win Probability by Round",
        color="white",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Round", color="#aaaaaa", fontsize=10)
    ax.set_ylabel("Win Probability (%)", color="#aaaaaa", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(
        ["Pre-\ntournament"] + [f"After\nR{r}" for r in x[1:]],
        color="#aaaaaa",
        fontsize=8,
    )
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.tick_params(colors="#aaaaaa")

    ax.set_ylim(0, max(max(v) for v in probs.values()) * 1.15)
    ax.set_xlim(x[0] - 0.3, x[-1] + 2.2)  # leave room for right-side labels

    for spine in ax.spines.values():
        spine.set_color("#333333")
    ax.grid(axis="y", color="#ffffff18", linewidth=0.6)
    ax.grid(axis="x", color="#ffffff0a", linewidth=0.4)

    # Subtitle: games played info
    total = len(players_list) * (len(players_list) - 1)
    played = sum(1 for r in x[1:])  # rough
    fig.text(
        0.5, 0.01,
        f"Monte Carlo simulation · {total} total games · ELO-weighted draw model",
        ha="center",
        color="#555555",
        fontsize=8,
    )

    plt.tight_layout(rect=[0, 0.03, 1, 1])

    if save:
        out = Path(__file__).parent / f"win_probability_{section}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"\nSaved: {out}")
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot win probability over rounds")
    parser.add_argument("section", choices=list(SECTIONS), help="open | womens")
    parser.add_argument("--sims", type=int, default=100_000,
                        help="Simulations per snapshot (default: 100000)")
    parser.add_argument("--save", action="store_true",
                        help="Save PNG instead of opening interactive window")
    args = parser.parse_args()

    print(f"\nSimulating {args.section} section after each round …")
    round_labels, probs = collect_probabilities(args.section, args.sims)

    print("\nBuilding plot …")
    make_plot(
        args.section,
        round_labels,
        probs,
        SECTIONS[args.section],
        save=args.save,
    )


if __name__ == "__main__":
    main()
