"""Build Causal Table for January 7, 2025 Rush-Hour Event (15:58:46 to 16:40:00).

Fleet: Strictly the 11 Track 0/4 Convoy Trains
Grid: 1-minute time series resolution (43 rows x 13 columns)
Output: data/causal_tables_jan2025/table1_20250107_rush_hour_11trains_1m.csv
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "causal_tables_jan2025"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAV_PATH = ROOT / "data" / "junction" / "traversals_202501.csv"
LONG_PATH = ROOT / "data" / "junction" / "junction_traversing_202501.csv"

# Exactly the 11 User-Specified Trains (The Track 0/4 Convoy)
TARGET_TRAINS = [
    "3786", "2435", "5137", "1937", "9240", "6587",
    "1737", "2136", "3687", "2339", "3987"
]


def secs(t) -> int:
    if not isinstance(t, str):
        return 0
    t = t.strip()
    if not t or ":" not in t:
        return 0
    try:
        parts = t.split(":")
        h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
        return h * 3600 + m * 60 + s
    except Exception:
        return 0


def format_secs(s: int) -> str:
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"


def build_january_causal_table():
    print("Loading January 2025 preprocessed datasets...")
    df_trav = pd.read_csv(TRAV_PATH)
    df_long = pd.read_csv(LONG_PATH)

    df_trav["train_no_str"] = df_trav["train_no"].astype(str)
    df_long["train_no_str"] = df_long["TRAIN_NO"].astype(str)

    # Focus on January 7, 2025 (07JAN2025) from 15:58:00 to 16:40:00 at 1-minute steps
    print("Building Table 1 (07-01-2025 from 15:58 to 16:40, 1-min grid, 11 trains)...")
    start_sec = 15 * 3600 + 58 * 60
    end_sec = 16 * 3600 + 40 * 60
    minute_grid = list(range(start_sec, end_sec + 60, 60))

    grid_df = pd.DataFrame({
        "date": "07JAN2025",
        "time": [format_secs(s) for s in minute_grid],
        "sec_grid": minute_grid
    })

    # Filter traversals for 07JAN2025 and target trains
    day_trav = df_trav[(df_trav["date"] == "07JAN2025") & (df_trav["train_no_str"].isin(TARGET_TRAINS))].copy()
    day_long = df_long[(df_long["DATDEP"] == "07JAN2025") & (df_long["train_no_str"].isin(TARGET_TRAINS))].copy()

    for t in TARGET_TRAINS:
        col_name = f"Train_{t}_delay"
        t_records = day_long[day_long["train_no_str"] == t].sort_values("REAL_TIME_DEP")
        
        if len(t_records) > 0:
            events = []
            for _, r in t_records.iterrows():
                sec_val = secs(str(r["REAL_TIME_DEP"]))
                delay_val = float(r["DELAY_DEP"]) if pd.notna(r["DELAY_DEP"]) else 0.0
                events.append((sec_val, delay_val))
            
            events.sort(key=lambda x: x[0])
            col_series = []
            for g_sec in minute_grid:
                matched_delay = None
                for ev_sec, ev_del in events:
                    if ev_sec <= g_sec:
                        matched_delay = ev_del
                    else:
                        break
                col_series.append(matched_delay)
            
            s_col = pd.Series(col_series)
            s_col = s_col.ffill().bfill()
            if s_col.isna().all():
                entry_del = day_trav[day_trav["train_no_str"] == t]["entry_delay"].values
                val = entry_del[0] if len(entry_del) > 0 else 0.0
                s_col = pd.Series([val] * len(grid_df))
            grid_df[col_name] = s_col.values
        else:
            entry_del = day_trav[day_trav["train_no_str"] == t]["entry_delay"].values
            val = entry_del[0] if len(entry_del) > 0 else 0.0
            grid_df[col_name] = val

    grid_df = grid_df.drop(columns=["sec_grid"])
    out_table = OUT_DIR / "table1_20250107_rush_hour_11trains_1m.csv"
    grid_df.to_csv(out_table, index=False)
    print(f"-> Saved: {out_table} ({grid_df.shape[0]} rows x {grid_df.shape[1]} cols)")


if __name__ == "__main__":
    build_january_causal_table()
