#!/usr/bin/env python3
"""FIDE Candidates 2026 – Monte Carlo Tournament Simulator.

Usage examples
--------------
# Show current standings
python main.py open standings

# Run simulation (200 k iterations)
python main.py open simulate

# Show full schedule with results
python main.py open schedule

# Show head-to-head matrix
python main.py open h2h

# Add a game result (round number is optional)
python main.py open add nakamura giri draw
python main.py open add caruana weiyi white --round 5
python main.py open add sindarov bluebaum black

# Remove a result (use if entered incorrectly)
python main.py open remove nakamura giri

# Clear ALL results (prompts for confirmation)
python main.py open clear

# Women's section
python main.py womens simulate
python main.py womens standings

# Adjust simulation parameters
python main.py open simulate --sims 500000 --draw-rate 0.60
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from config import SECTIONS
from tournament import Tournament
from simulation import simulate
from display import (
    print_standings,
    print_probabilities,
    print_schedule,
    print_head_to_head,
)

DATA_DIR = Path(__file__).parent / "data"


def parse_result(value: str) -> float:
    """Accept '1'/'white'/'w', '0.5'/'draw'/'d'/'½', '0'/'black'/'b'."""
    v = value.strip().lower()
    if v in ("1", "1.0", "white", "w", "win"):
        return 1.0
    if v in ("0.5", "draw", "d", "½", "half"):
        return 0.5
    if v in ("0", "0.0", "black", "b", "loss"):
        return 0.0
    raise argparse.ArgumentTypeError(
        f"Invalid result '{value}'. Use: white/w/1  draw/d/½  black/b/0"
    )


def results_path(section: str) -> Path:
    return DATA_DIR / f"{section}_results.json"


def load_tournament(section: str) -> Tournament:
    if section not in SECTIONS:
        print(f"Unknown section '{section}'. Choose from: {', '.join(SECTIONS)}")
        sys.exit(1)
    return Tournament(SECTIONS[section], results_path(section))


# ---------------------------------------------------------------------------
# Sub-command handlers
# ---------------------------------------------------------------------------

def cmd_standings(args: argparse.Namespace) -> None:
    t = load_tournament(args.section)
    print_standings(t)


def cmd_simulate(args: argparse.Namespace) -> None:
    t = load_tournament(args.section)
    n = args.sims
    dr = args.draw_rate

    print(f"\nRunning {n:,} simulations (draw rate base={dr:.2f}) …", end="", flush=True)
    results = simulate(t, n_sims=n, base_draw=dr)
    print(" done.\n")

    print_standings(t)
    print_probabilities(t, results)


def cmd_schedule(args: argparse.Namespace) -> None:
    t = load_tournament(args.section)
    print_schedule(t)


def cmd_h2h(args: argparse.Namespace) -> None:
    t = load_tournament(args.section)
    print_head_to_head(t)


def cmd_add(args: argparse.Namespace) -> None:
    t = load_tournament(args.section)
    white = args.white.lower()
    black = args.black.lower()
    result = parse_result(args.result)

    # Accept last-name prefix matching as well as exact ID matching
    def resolve(token: str) -> str:
        token = token.lower()
        if token in t.players:
            return token
        # Fuzzy: try matching against name (last word) or full name
        matches = [
            pid for pid, p in t.players.items()
            if token in p.name.lower() or token == pid
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            names = [t.players[m].name for m in matches]
            raise ValueError(f"Ambiguous player '{token}': {names}")
        raise ValueError(
            f"Unknown player '{token}'. "
            f"Valid IDs: {', '.join(sorted(t.players))}"
        )

    try:
        white_id = resolve(white)
        black_id = resolve(black)
        t.add_result(white_id, black_id, result, round_num=args.round)
        w_name = t.players[white_id].name
        b_name = t.players[black_id].name
        sym = {1.0: "1–0", 0.5: "½–½", 0.0: "0–1"}[result]
        rnd = f" (round {args.round})" if args.round else ""
        print(f"  Added: {w_name} {sym} {b_name}{rnd}")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


def cmd_remove(args: argparse.Namespace) -> None:
    t = load_tournament(args.section)
    try:
        t.remove_result(args.white.lower(), args.black.lower())
        print(f"  Removed result for {args.white} (W) vs {args.black} (B).")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


def cmd_clear(args: argparse.Namespace) -> None:
    t = load_tournament(args.section)
    count = t.games_played()
    if count == 0:
        print("No results to clear.")
        return
    confirm = input(
        f"This will delete all {count} recorded results for '{args.section}'. "
        "Type YES to confirm: "
    )
    if confirm.strip() == "YES":
        t.clear()
        print("All results cleared.")
    else:
        print("Cancelled.")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FIDE Candidates 2026 Monte Carlo Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "section",
        choices=list(SECTIONS),
        help="Tournament section: open | womens",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # standings
    sub.add_parser("standings", help="Show current standings")

    # simulate
    sim_p = sub.add_parser("simulate", aliases=["sim"], help="Run Monte Carlo simulation")
    sim_p.add_argument("--sims", type=int, default=200_000,
                       help="Number of simulations (default: 200000)")
    sim_p.add_argument("--draw-rate", type=float, default=0.58, dest="draw_rate",
                       help="Base draw probability at equal ELO (default: 0.58)")

    # schedule
    sub.add_parser("schedule", aliases=["sched"], help="Show full schedule with results")

    # head-to-head
    sub.add_parser("h2h", help="Show head-to-head results matrix")

    # add result
    add_p = sub.add_parser("add", help="Add a game result")
    add_p.add_argument("white", help="White player ID or name fragment")
    add_p.add_argument("black", help="Black player ID or name fragment")
    add_p.add_argument("result", help="Result: white/w/1  draw/d/½  black/b/0")
    add_p.add_argument("--round", type=int, default=0, metavar="N",
                       help="Round number (optional, informational)")

    # remove result
    rem_p = sub.add_parser("remove", aliases=["rm"], help="Remove a recorded result")
    rem_p.add_argument("white", help="White player ID")
    rem_p.add_argument("black", help="Black player ID")

    # clear
    sub.add_parser("clear", help="Clear ALL results (interactive confirmation required)")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    command_map = {
        "standings": cmd_standings,
        "simulate":  cmd_simulate,
        "sim":       cmd_simulate,
        "schedule":  cmd_schedule,
        "sched":     cmd_schedule,
        "h2h":       cmd_h2h,
        "add":       cmd_add,
        "remove":    cmd_remove,
        "rm":        cmd_remove,
        "clear":     cmd_clear,
    }

    handler = command_map.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
