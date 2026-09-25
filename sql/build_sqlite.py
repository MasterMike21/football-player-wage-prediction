from __future__ import annotations

from pathlib import Path
import sqlite3

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "cleaned_players.csv"
DB_PATH = ROOT / "artifacts" / "football_wages.db"

def main():
    import pandas as pd

    df = pd.read_csv(CSV_PATH)

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)

    df = df.rename(columns={
        "Wage": "wage",
        "Age": "age",
        "Club": "club",
        "League": "league",
        "Nation": "nation",
        "Position": "position",
        "Apps": "apps",
        "Caps": "caps",
        "Caps_per_App": "caps_per_app",
        "Apps_per_Year": "apps_per_year",
        "International_Exposure": "international_exposure",
    })

    df.to_sql("players", conn, if_exists="replace", index=False)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_players_league ON players(league);
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_players_club ON players(club);
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_players_position ON players(position);
    """)
    conn.commit()
    conn.close()

    print(f"SQLite database created: {DB_PATH}")

if __name__ == "__main__":
    main()
