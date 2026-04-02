"""Player configurations for the 2026 FIDE Candidates Tournament.

ELO ratings: FIDE March 2026 list.
Source: https://candidates2026.fide.com

To add or update players, edit the lists below.
"""

from __future__ import annotations

from tournament import Player

# ---------------------------------------------------------------------------
# 2026 FIDE Candidates – Open Section
# ---------------------------------------------------------------------------
OPEN_2026_PLAYERS: list[Player] = [
    Player("nakamura",  "Hikaru Nakamura",          "USA",         2810),
    Player("caruana",   "Fabiano Caruana",           "USA",         2795),
    Player("sindarov",  "Javokhir Sindarov",         "Uzbekistan",  2745),
    Player("pragg",     "R. Praggnanandhaa",         "India",       2741),
    Player("giri",      "Anish Giri",                "Netherlands", 2753),
    Player("weiyi",     "Wei Yi",                    "China",       2754),
    Player("bluebaum",  "Matthias Blübaum",          "Germany",     2698),
    Player("esipenko",  "Andrey Esipenko",           "FIDE",        2698),
]

# ---------------------------------------------------------------------------
# 2026 FIDE Women's Candidates
# ---------------------------------------------------------------------------
WOMENS_2026_PLAYERS: list[Player] = [
    Player("zhujiner",     "Zhu Jiner",              "China",       2578),
    Player("tanzhongyi",   "Tan Zhongyi",            "China",       2535),
    Player("goryachkina",  "Aleksandra Goryachkina", "FIDE",        2534),
    Player("muzychuk",     "Anna Muzychuk",          "Ukraine",     2522),
    Player("assaubayeva",  "Bibisara Assaubayeva",   "Kazakhstan",  2516),
    Player("lagno",        "Kateryna Lagno",         "FIDE",        2508),
    Player("deshmukh",     "Divya Deshmukh",         "India",       2497),
    Player("vaishali",     "Vaishali Rameshbabu",    "India",       2470),
]

SECTIONS: dict[str, list[Player]] = {
    "open":   OPEN_2026_PLAYERS,
    "womens": WOMENS_2026_PLAYERS,
}
