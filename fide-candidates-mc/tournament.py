"""Tournament state management: players, schedule, results persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


@dataclass
class Player:
    id: str          # short slug, e.g. "nakamura"
    name: str        # display name
    country: str
    elo: float


@dataclass
class GameResult:
    white: str       # player id
    black: str       # player id
    result: float    # 1.0 = white wins, 0.5 = draw, 0.0 = black wins
    round: int = 0   # informational; 0 = not specified


class Tournament:
    """Manages a double round-robin tournament with persistent results."""

    def __init__(self, players: list[Player], results_path: Path):
        self.players: dict[str, Player] = {p.id: p for p in players}
        self.player_ids: list[str] = [p.id for p in players]
        self.results_path = Path(results_path)
        self.results: list[GameResult] = []
        self._load()

    # ------------------------------------------------------------------ I/O --

    def _load(self) -> None:
        if self.results_path.exists():
            with open(self.results_path, encoding="utf-8") as f:
                data = json.load(f)
            self.results = [GameResult(**g) for g in data]

    def save(self) -> None:
        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.results_path, "w", encoding="utf-8") as f:
            json.dump([asdict(g) for g in self.results], f, indent=2)

    # ----------------------------------------------------------- mutations --

    def add_result(
        self,
        white: str,
        black: str,
        result: float,
        round_num: int = 0,
    ) -> None:
        """Record a game result.  Raises ValueError on bad input or duplicate."""
        white, black = white.lower(), black.lower()
        if white not in self.players:
            raise ValueError(f"Unknown player: '{white}'")
        if black not in self.players:
            raise ValueError(f"Unknown player: '{black}'")
        if white == black:
            raise ValueError("White and black must be different players.")
        if result not in (0.0, 0.5, 1.0):
            raise ValueError(f"Result must be 0.0, 0.5, or 1.0; got {result!r}")
        played = self._played_pairs()
        if (white, black) in played:
            raise ValueError(
                f"{self.players[white].name} (W) vs {self.players[black].name} (B) "
                "already recorded."
            )
        self.results.append(GameResult(white, black, result, round_num))
        self.save()

    def remove_result(self, white: str, black: str) -> None:
        """Delete a previously recorded result."""
        white, black = white.lower(), black.lower()
        before = len(self.results)
        self.results = [
            r for r in self.results
            if not (r.white == white and r.black == black)
        ]
        if len(self.results) == before:
            raise ValueError(
                f"No recorded result for {white} (W) vs {black} (B)."
            )
        self.save()

    def clear(self) -> None:
        """Remove all results."""
        self.results = []
        self.save()

    # ------------------------------------------------------------ queries --

    def _played_pairs(self) -> set[tuple[str, str]]:
        return {(r.white, r.black) for r in self.results}

    def standings(self) -> dict[str, float]:
        """Return current points for each player."""
        scores: dict[str, float] = {pid: 0.0 for pid in self.player_ids}
        for r in self.results:
            scores[r.white] += r.result
            scores[r.black] += 1.0 - r.result
        return scores

    def wins(self) -> dict[str, int]:
        """Return number of decisive wins for each player."""
        w: dict[str, int] = {pid: 0 for pid in self.player_ids}
        for r in self.results:
            if r.result == 1.0:
                w[r.white] += 1
            elif r.result == 0.0:
                w[r.black] += 1
        return w

    def remaining_games(self) -> list[tuple[str, str]]:
        """All (white, black) ordered pairs not yet played."""
        played = self._played_pairs()
        return [
            (w, b)
            for w in self.player_ids
            for b in self.player_ids
            if w != b and (w, b) not in played
        ]

    def games_played(self) -> int:
        return len(self.results)

    def total_games(self) -> int:
        n = len(self.players)
        return n * (n - 1)  # double round-robin

    def rounds_played(self) -> int:
        """Estimate completed rounds (floor of games_played / games_per_round)."""
        n = len(self.players)
        games_per_round = n // 2
        return self.games_played() // games_per_round

    def result_for(self, white: str, black: str) -> Optional[float]:
        """Return recorded result or None if not yet played."""
        for r in self.results:
            if r.white == white and r.black == black:
                return r.result
        return None

    # ---------------------------------------------- schedule generation --

    def schedule(self) -> list[list[tuple[str, str]]]:
        """Generate canonical double round-robin schedule using the circle method.

        The official pairings may differ; this is a reference schedule used
        for display.  Results are stored by (white, black) pair regardless.
        """
        ids = self.player_ids
        n = len(ids)
        fixed = ids[-1]
        rotating = list(ids[:-1])

        first_half: list[list[tuple[str, str]]] = []
        for rnd in range(n - 1):
            pairs: list[tuple[str, str]] = []
            if rnd % 2 == 0:
                pairs.append((rotating[0], fixed))
            else:
                pairs.append((fixed, rotating[0]))
            for j in range(1, n // 2):
                p1, p2 = rotating[j], rotating[n - 1 - j]
                if (rnd + j) % 2 == 0:
                    pairs.append((p1, p2))
                else:
                    pairs.append((p2, p1))
            first_half.append(pairs)
            rotating = [rotating[-1]] + rotating[:-1]

        second_half = [[(b, a) for a, b in rnd_pairs] for rnd_pairs in first_half]
        return first_half + second_half
