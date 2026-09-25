-- SQLite schema for Football Player Wage Intelligence

DROP TABLE IF EXISTS players;

CREATE TABLE players (
    player_id INTEGER PRIMARY KEY AUTOINCREMENT,
    wage REAL NOT NULL,
    age INTEGER NOT NULL,
    club TEXT NOT NULL,
    league TEXT NOT NULL,
    nation TEXT NOT NULL,
    position TEXT NOT NULL,
    apps INTEGER NOT NULL,
    caps INTEGER NOT NULL,
    caps_per_app REAL NOT NULL,
    apps_per_year REAL NOT NULL,
    international_exposure REAL NOT NULL
);

-- The Python build script loads cleaned_players.csv into this table.
