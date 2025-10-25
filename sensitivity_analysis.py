# sensitivity_oaat.py
# -*- coding: utf-8 -*-
"""
One-at-a-time (±10%) local sensitivity analysis for your hydration model.
- Imports your model module (default path below).
- Runs baseline once, then perturbs each selected parameter by ±10% (or additive if baseline=0).
- Computes paper-ready metrics and exports:
    * CSV: sensitivity_<TARGET>.csv
    * PNG: sensitivity_radar_<TARGET>.png

USAGE (example):
    python sensitivity_oaat.py \
        --model /mnt/data/4fe4679e-12c6-49c6-9702-c77ec0909d4c.py \
        --target CSH_peak_rate \
        --rel-step 0.10

Notes:
- Your model is expected to define CONFIG and main(cfg) and to write results to CONFIG["out_dir"].
- We read activities from CONFIG["activities_csv"] to compute pH; ensure it exists.
"""

import argparse
import importlib.util
import sys
from pathlib import Path
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Defaults (edit if you like)
# ----------------------------
DEFAULT_MODEL_PATH = "/mnt/data/4fe4679e-12c6-49c6-9702-c77ec0909d4c.py"

# Parameters to analyze (must exist in model CONFIG)
PARAM_LIST = [
    "A_geom",         # 比表面积
    "k_seed",         # C-S-H 成核/增长速率因子
    "k_AFt",          # AFt 生成速率
    "k_AFm",          # AFm 生成速率
    "theta0",         # 覆盖度基值
    "k_agg_I",        # 离子强度致团聚因子
    "K_CO3", "K_SO4", # 竞争吸附常数
    "alpha_CO3", "beta_SO4",
    "k_Mg",
    "beta_delta", "delta_SI_star",
    "K_cit",          # 柠檬酸抑制
]

# If baseline value==0, multiplicative ±10% is meaningless — use additive step scale for such params
PARAM_SCALES = {
    "theta0": 0.05,
    "k_agg_I": 0.2,
    "k_Mg": 0.2,
    "K_cit": 1.0,
}

# Metrics to compute from model outputs
OUTPUT_VARS = ["pH_mean", "AFt/AFm_final", "CSH_peak_rate"]

# ----------------------------
# Model import helper
# ----------------------------
def import_model(model_path: str):
    spec = importlib.util.spec_from_file_location("hydration_model", model_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["hydration_model"] = mod
    spec.loader.exec_module(mod)
    return mod

# ----------------------------
# Metrics & run helpers
# ----------------------------
def clone_config(mod):
    return {k: v for k, v in mod.CONFIG.items()}

def run_model_once(mod, cfg: dict) -> pd.DataFrame:
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    mod.main(cfg)  # model writes time_series_results.csv into out_dir
    ts_path = out_dir / "time_series_results.csv"
    if not ts_path.exists():
        raise FileNotFoundError(f"[sensitivity] Missing results: {ts_path}")
    return pd.read_csv(ts_path)

def compute_metrics(timeseries_df: pd.DataFrame, cfg: dict) -> dict:
    """
    - pH_mean: from activities CSV using pH=14+log10(a_OH), averaged over time
    - AFt/AFm_final: n_AFt / (n_HC + n_MC) at final time
    - CSH_peak_rate: max(R_A)
    """
    # pH from activities
    acts = pd.read_csv(cfg["activities_csv"])
    a_OH_col = cfg["csv_cols"]["a_OH"]
    if a_OH_col not in acts.columns:
        raise ValueError(f"[sensitivity] a_OH column '{a_OH_col}' not found in activities CSV.")
    a_OH = acts[a_OH_col].to_numpy(dtype=float)
    pH_series = 14.0 + np.log10(np.clip(a_OH, 1e-12, None))
    pH_mean = float(np.mean(pH_series))

    # time series metrics
    n_AFt = timeseries_df["n_AFt"].to_numpy()
    n_HC  = timeseries_df["n_HC"].to_numpy()
    n_MC  = timeseries_df["n_MC"].to_numpy()
    RA    = timeseries_df["R_A"].to_numpy()

    AFm_sum = max(float(n_HC[-1] + n_MC[-1]), 1e-12)
    aft_over_afm = float(n_AFt[-1]) / AFm_sum
    csh_peak = float(np.max(RA))
    return {"pH_mean": pH_mean, "AFt/AFm_final": aft_over_afm, "CSH_peak_rate": csh_peak}

def perturb_param(baseline: float, rel_step: float, direction: int, scale: float = 1.0) -> float:
    """
    direction: +1 or -1
    multiplicative if baseline != 0; additive with given 'scale' if baseline == 0
    """
    if abs(baseline) > 1e-12:
        return baseline * (1.0 + direction * rel_step)
    else:
        return baseline + direction * rel_step * scale

# ----------------------------
# Sensitivity core
# ----------------------------
def sensitivity_oaat(mod, target: str, rel_step: float, analysis_dir: Path):
    cfg0 = clone_config(mod)

    # Baseline
    ts0 = run_model_once(mod, cfg0)
    base_metrics = compute_metrics(ts0, cfg0)
    y0 = base_metrics[target]

    rows = []
    for p in PARAM_LIST:
        v0 = cfg0[p]
        scale = PARAM_SCALES.get(p, 1.0)
        # +
        cfg_plus = clone_config(mod)
        cfg_plus.update(cfg0)
        cfg_plus[p] = perturb_param(v0, rel_step, +1, scale)
        y_plus = compute_metrics(run_model_once(mod, cfg_plus), cfg_plus)[target]
        # -
        cfg_minus = clone_config(mod)
        cfg_minus.update(cfg0)
        cfg_minus[p] = perturb_param(v0, rel_step, -1, scale)
        y_minus = compute_metrics(run_model_once(mod, cfg_minus), cfg_minus)[target]

        denom = 2.0 * rel_step
        S = ((y_plus - y_minus) / denom) / (y0 if y0 != 0 else 1.0)
        rows.append({"parameter": p, "sensitivity": S})

    df = pd.DataFrame(rows)
    # save CSV
    out_csv = analysis_dir / f"sensitivity_{target}.csv"
    df.to_csv(out_csv, index=False)

    # plot radar (normalized for visual only)
    plot_radar(df, target, analysis_dir / f"sensitivity_radar_{target}.png")

    # also save baseline metrics for reference
    pd.DataFrame([base_metrics]).to_csv(analysis_dir / "baseline_metrics.csv", index=False)
    return out_csv

def plot_radar(df: pd.DataFrame, target: str, outfile: Path):
    # keep order of PARAM_LIST
    df = df.set_index("parameter").reindex(PARAM_LIST).reset_index()
    labels = df["parameter"].tolist()
    values = df["sensitivity"].to_numpy()
    max_abs = np.max(np.abs(values)) if np.max(np.abs(values)) > 0 else 1.0
    values_norm = values / max_abs

    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    values_closed = np.concatenate([values_norm, values_norm[:1]])
    angles_closed = angles + angles[:1]

    fig = plt.figure(figsize=(7,7))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles_closed, values_closed, linewidth=2)
    ax.fill(angles_closed, values_closed, alpha=0.1)
    ax.set_xticks(angles); ax.set_xticklabels(labels)
    ax.set_ylim(-1.0, 1.0)
    ax.set_title(f"Sensitivity (target={target})")
    plt.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=200)
    plt.close(fig)

# ----------------------------
# CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="Path to your model .py")
    ap.add_argument("--target", type=str, default="CSH_peak_rate", choices=["pH_mean","AFt/AFm_final","CSH_peak_rate"])
    ap.add_argument("--rel-step", type=float, default=0.10, help="Relative step for ± perturbation (default 0.10)")
    ap.add_argument("--analysis-dir", type=str, default="./analysis_outputs", help="Folder to write CSV/figures")
    args = ap.parse_args()

    mod = import_model(args.model)
    analysis_dir = Path(args.analysis_dir)
    csv_path = sensitivity_oaat(mod, args.target, args.rel_step, analysis_dir)
    print(f"[OK] Sensitivity CSV: {csv_path}")
    print(f"[OK] Figure: {analysis_dir / ('sensitivity_radar_' + args.target + '.png')}")

if __name__ == "__main__":
    main()
