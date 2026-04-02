"""Terminal display utilities: standings, probabilities, schedule tables."""

from __future__ import annotations

import sys
import io

# Reconfigure stdout to UTF-8 on Windows so box-drawing chars render correctly
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from tournament import Tournament

# ANSI colors (gracefully disabled on terminals that don't support them)
_BOLD   = "\033[1m"
_CYAN   = "\033[96m"
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_RESET  = "\033[0m"

RESULT_SYMBOL = {1.0: "1–0", 0.5: "½–½", 0.0: "0–1"}


def _bar(prob: float, width: int = 20) -> str:
    filled = round(prob * width)
    return "█" * filled + "░" * (width - filled)


def print_header(title: str) -> None:
    line = "─" * (len(title) + 4)
    print(f"\n{_BOLD}{_CYAN}┌{line}┐{_RESET}")
    print(f"{_BOLD}{_CYAN}│  {title}  │{_RESET}")
    print(f"{_BOLD}{_CYAN}└{line}┘{_RESET}\n")


def print_standings(tournament: Tournament) -> None:
    """Print current standings table."""
    print_header("CURRENT STANDINGS")
    scores = tournament.standings()
    wins = tournament.wins()
    played = {
        pid: sum(1 for r in tournament.results if r.white == pid or r.black == pid)
        for pid in tournament.player_ids
    }
    n_total = tournament.total_games() // len(tournament.player_ids)  # games per player

    ranked = sorted(
        tournament.player_ids,
        key=lambda p: (-scores[p], -wins[p], -tournament.players[p].elo),
    )

    col = f"{'#':<3} {'Player':<24} {'Country':<14} {'ELO':<6} {'Score':<8} {'Wins':<6} {'Played':<10}"
    print(_BOLD + col + _RESET)
    print("─" * len(col))

    prev_score = None
    rank = 0
    for i, pid in enumerate(ranked):
        p = tournament.players[pid]
        s = scores[pid]
        if s != prev_score:
            rank = i + 1
            prev_score = s
        gp = played[pid]
        score_str = f"{s:.1f}/{gp}"
        color = _GREEN if rank == 1 else (_YELLOW if rank <= 3 else "")
        print(
            f"{color}{rank:<3} {p.name:<24} {p.country:<14} {p.elo:<6.0f} "
            f"{score_str:<8} {wins[pid]:<6} {gp}/{n_total}{_RESET}"
        )

    total = tournament.total_games()
    played_total = tournament.games_played()
    remaining = total - played_total
    print(
        f"\n  Games played: {played_total}/{total}  "
        f"({remaining} remaining, ~{remaining // (len(tournament.players) // 2)} rounds left)"
    )


def print_probabilities(tournament: Tournament, sim_results: dict[str, dict]) -> None:
    """Print win-probability table."""
    print_header("MONTE CARLO WIN PROBABILITIES")
    scores = tournament.standings()
    ranked = sorted(
        tournament.player_ids,
        key=lambda p: -sim_results[p]["win_prob"],
    )

    col = (
        f"{'#':<3} {'Player':<24} {'ELO':<6} {'Score':<7} "
        f"{'Win %':<8} {'Proj. Score (Q10–Q90)':<26} {'Probability'}"
    )
    print(_BOLD + col + _RESET)
    print("─" * (len(col) + 22))  # +22 for bar width

    for rank, pid in enumerate(ranked, 1):
        p = tournament.players[pid]
        r = sim_results[pid]
        prob = r["win_prob"]
        gp = sum(1 for g in tournament.results if g.white == pid or g.black == pid)
        score_str = f"{scores[pid]:.1f}/{gp}"
        proj_str = f"{r['q10_pts']:.1f}–{r['median_pts']:.1f}–{r['q90_pts']:.1f}"

        if prob >= 0.50:
            color = _GREEN
        elif prob >= 0.15:
            color = _YELLOW
        elif prob >= 0.01:
            color = ""
        else:
            color = _RED

        print(
            f"{color}{rank:<3} {p.name:<24} {p.elo:<6.0f} {score_str:<7} "
            f"{prob*100:>6.1f}%  {proj_str:<26} {_bar(prob)}{_RESET}"
        )


def print_schedule(tournament: Tournament) -> None:
    """Print full double round-robin schedule with results."""
    print_header("SCHEDULE & RESULTS")
    sched = tournament.schedule()
    players = tournament.players

    for rnd_idx, pairs in enumerate(sched, 1):
        games_in_round = len(pairs)
        completed = sum(
            1 for w, b in pairs if tournament.result_for(w, b) is not None
        )
        status = f"({completed}/{games_in_round} played)"
        print(f"  {_BOLD}Round {rnd_idx:>2}{_RESET}  {status}")
        for w_id, b_id in pairs:
            result = tournament.result_for(w_id, b_id)
            w_name = players[w_id].name
            b_name = players[b_id].name
            if result is not None:
                sym = RESULT_SYMBOL[result]
                if result == 1.0:
                    line = f"    {_GREEN}{w_name:<22}{_RESET} {sym}  {b_name}"
                elif result == 0.0:
                    line = f"    {w_name:<22} {sym}  {_GREEN}{b_name}{_RESET}"
                else:
                    line = f"    {w_name:<22} {sym}  {b_name}"
            else:
                line = f"    {w_name:<22}  vs  {b_name}"
            print(line)
        print()


def print_head_to_head(tournament: Tournament) -> None:
    """Print head-to-head results matrix."""
    print_header("HEAD-TO-HEAD MATRIX  (row = white)")
    ids = tournament.player_ids
    players = tournament.players
    short = {pid: players[pid].name.split()[-1][:10] for pid in ids}

    # Header row
    header = f"{'':20}" + "".join(f"{short[pid]:>12}" for pid in ids)
    print(_BOLD + header + _RESET)
    print("─" * len(header))

    for w_id in ids:
        row = f"{players[w_id].name:<20}"
        for b_id in ids:
            if w_id == b_id:
                row += f"{'—':>12}"
            else:
                r = tournament.result_for(w_id, b_id)
                if r is None:
                    row += f"{'·':>12}"
                else:
                    row += f"{RESULT_SYMBOL[r]:>12}"
        print(row)
