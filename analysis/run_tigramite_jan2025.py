"""Run Official Tigramite PCMCI with RobustParCorr and Forward-Only Link Assumptions.

Scope:
- Target Day: January 7, 2025 (15:58:46 to 16:40:00)
- Fleet: The 11 Track 0/4 Convoy Trains
- Library: tigramite (official implementation)
- Independence Test: RobustParCorr(significance='analytic')
- Output: analysis/causal_discovery_results_jan2025/tigramite_pcmci_results_tau1_10.json
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from tigramite import data_processing as pp
from tigramite.independence_tests.robust_parcorr import RobustParCorr
from tigramite.pcmci import PCMCI

ROOT = Path(__file__).resolve().parents[1]
TABLE_PATH = ROOT / "data" / "causal_tables_jan2025" / "table1_20250107_rush_hour_11trains_1m.csv"
OUT_DIR = ROOT / "analysis" / "causal_discovery_results_jan2025"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Exact departure timestamps from Infrabel 07JAN2025 dataset
T_DEP = {
    "3786": 15 * 3600 + 55 * 60 + 2,
    "2435": 15 * 3600 + 58 * 60 + 26,
    "5137": 16 * 3600 + 0 * 60 + 44,
    "1937": 16 * 3600 + 4 * 60 + 3,
    "9240": 16 * 3600 + 7 * 60 + 50,
    "6587": 16 * 3600 + 11 * 60 + 24,
    "1737": 16 * 3600 + 13 * 60 + 59,
    "2136": 16 * 3600 + 18 * 60 + 13,
    "3687": 16 * 3600 + 22 * 60 + 31,
    "2339": 16 * 3600 + 27 * 60 + 36,
    "3987": 16 * 3600 + 33 * 60 + 17,
}


def run_tigramite_pcmci(tau_max: int = 10, alpha_level: float = 0.05) -> dict:
    print(f"Loading January 7, 2025 rush-hour data for 11 convoy trains...")
    df = pd.read_csv(TABLE_PATH)
    train_cols = [c for c in df.columns if c.startswith("Train_")]
    data = df[train_cols].values
    var_names = [c.replace("Train_", "").replace("_delay", "") for c in train_cols]
    N = len(var_names)

    dataframe = pp.DataFrame(data, datatime=np.arange(len(data)), var_names=var_names)
    robust_test = RobustParCorr(significance="analytic")
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=robust_test, verbosity=0)

    # Build Tigramite link_assumptions: allow only forward physical causality
    link_assumptions = {}
    for j in range(N):
        link_assumptions[j] = {}
        tgt_t = var_names[j]
        for i in range(N):
            src_t = var_names[i]
            if i == j or T_DEP[src_t] < T_DEP[tgt_t]:
                for tau in range(1, tau_max + 1):
                    link_assumptions[j][(i, -tau)] = "-?>"

    print(f"Running Tigramite PCMCI with tau_max={tau_max}, alpha_level={alpha_level} and link_assumptions...")
    results = pcmci.run_pcmci(
        tau_max=tau_max,
        link_assumptions=link_assumptions,
        pc_alpha=alpha_level,
        alpha_level=alpha_level
    )

    edges = []
    for j in range(N):
        for i in range(N):
            for tau in range(1, tau_max + 1):
                link_type = results["graph"][i, j, tau]
                if link_type == "-->":
                    val = float(results["val_matrix"][i, j, tau])
                    p = float(results["p_matrix"][i, j, tau])
                    edges.append({
                        "source": f"Train_{var_names[i]}_delay",
                        "target": f"Train_{var_names[j]}_delay",
                        "source_name": f"Train {var_names[i]}",
                        "target_name": f"Train {var_names[j]}",
                        "lag": tau,
                        "partial_corr": val,
                        "p_value": p,
                        "is_self_autocorrelation": (i == j)
                    })

    edges.sort(key=lambda e: abs(e["partial_corr"]), reverse=True)
    cross_edges = [e for e in edges if not e["is_self_autocorrelation"]]
    print(f"-> Tigramite execution complete: {len(edges)} total links ({len(cross_edges)} cross-variable links)")

    out_data = {
        "algorithm": "Tigramite PCMCI",
        "cond_ind_test": "RobustParCorr (analytic)",
        "methodology": "Solution A (Tigramite link_assumptions)",
        "tau_max": tau_max,
        "alpha_level": alpha_level,
        "n_samples": len(df),
        "variables": [f"Train_{v}_delay" for v in var_names],
        "var_names": var_names,
        "edges": edges
    }

    out_json = OUT_DIR / "tigramite_pcmci_results_tau1_10.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)
    print(f"-> Saved: {out_json}")
    return out_data


if __name__ == "__main__":
    run_tigramite_pcmci(tau_max=10, alpha_level=0.05)
