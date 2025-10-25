# stability_monte_carlo.py
# -*- coding: utf-8 -*-
"""
Monte Carlo (±10%) stability / uncertainty propagation for your hydration model.
- Randomizes ALL selected parameters together within ±range around baseline (multiplicative; additive if baseline=0).
- Runs the model for each sample.
- Computes per-metric mean and 95% CI; exports:
    * CSV: monte_carlo_samples.csv (long-form), monte_carlo_ci.csv (summary)
    * PNG: stability_errorbars.png (mean±95%CI), stability_boxplot.png (distribution shape)

USAGE (example):
    python stability_monte_carlo.py \
        --model /mnt/data/4fe4679e-12c6-49c6-9702-c77ec0909d4c.py \
        --samples 200 --rel-range 0.10
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

DEFAULT_MODEL_PATH = "/mnt/data/4fe4679e-12c6-49c6-9702-c77ec0909d4c.py"

PARAM_LIST = [
    "A_geom", "k_seed", "k_AFt", "k_AFm", "theta0",
    "k_agg_I", "K_CO3", "K_SO4", "alpha_CO3", "beta_SO4",
    "k_Mg", "beta_delta", "delta_SI_star", "K_cit",
]

PARAM_SCALES = {
    "theta0": 0.05,
    "k_agg_I": 0.2,
    "k_Mg": 0.2,
    "K_cit": 1.0,
}

OUTPUT_VARS = ["pH_mean", "AFt/AFm_final", "CSH_peak_rate"]

# ------------- model import -------------
def import_model(model_path: str):
    spec = importlib.util.spec_from_file_location("hydration_model", model_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["hydration_model"] = mod
    spec.loader.exec_module(mod)
    return mod

# ------------- helpers ------------------
def clone_config(mod):
    return {k: v for k, v in mod.CONFIG.items()}

def run_model_once(mod, cfg: dict) -> pd.DataFrame:
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    mod.main(cfg)
    ts_path = out_dir / "time_series_results.csv"
    if not ts_path.exists():
        raise FileNotFoundError(f"[stability] Missing results: {ts_path}")
    return pd.read_csv(ts_path)

def compute_metrics(timeseries_df: pd.DataFrame, cfg: dict) -> dict:
    acts = pd.read_csv(cfg["activities_csv"])
    a_OH_col = cfg["csv_cols"]["a_OH"]
    if a_OH_col not in acts.columns:
        raise ValueError(f"[stability] a_OH column '{a_OH_col}' not found in activities CSV.")
    a_OH = acts[a_OH_col].to_numpy(dtype=float)
    pH_series = 14.0 + np.log10(np.clip(a_OH, 1e-12, None))
    pH_mean = float(np.mean(pH_series))

    n_AFt = timeseries_df["n_AFt"].to_numpy()
    n_HC  = timeseries_df["n_HC"].to_numpy()
    n_MC  = timeseries_df["n_MC"].to_numpy()
    RA    = timeseries_df["R_A"].to_numpy()
    AFm_sum = max(float(n_HC[-1] + n_MC[-1]), 1e-12)
    aft_over_afm = float(n_AFt[-1]) / AFm_sum
    csh_peak = float(np.max(RA))
    return {"pH_mean": pH_mean, "AFt/AFm_final": aft_over_afm, "CSH_peak_rate": csh_peak}

def randomize_param(baseline: float, rel_range: float, scale: float) -> float:
    """
    multiplicative uniform in [1-rel, 1+rel] if baseline!=0;
    additive uniform in [-rel*scale, +rel*scale] if baseline==0.
    """
    if abs(baseline) > 1e-12:
        r = (1 - rel_range) + random.random() * (2 * rel_range)
        return baseline * r
    else:
        delta = (random.random() * 2 * rel_range - rel_range) * scale
        return baseline + delta

# ------------- Monte Carlo core ---------
def monte_carlo(mod, samples: int, rel_range: float, analysis_dir: Path):
    base_cfg = clone_config(mod)
    recs = []
    for s in range(samples):
        cfg = clone_config(mod)
        cfg.update(base_cfg)
        for p in PARAM_LIST:
            v0 = base_cfg[p]
            scale = PARAM_SCALES.get(p, 1.0)
            cfg[p] = randomize_param(v0, rel_range, scale)
        ts = run_model_once(mod, cfg)
        metrics = compute_metrics(ts, cfg)
        for k, val in metrics.items():
            recs.append({"sample": s, "var": k, "value": float(val)})

    df_long = pd.DataFrame(recs)
    out_csv = analysis_dir / "monte_carlo_samples.csv"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    df_long.to_csv(out_csv, index=False)

    ci_df = ci_summary(df_long, ci=0.95)
    ci_df.to_csv(analysis_dir / "monte_carlo_ci.csv", index=False)

    plot_errorbars(ci_df, analysis_dir / "stability_errorbars.png")
    plot_boxplots(df_long, analysis_dir / "stability_boxplot.png")
    return out_csv

def ci_summary(df_long: pd.DataFrame, ci: float = 0.95) -> pd.DataFrame:
    z = 1.96  # 95% two-sided normal approx
    rows = []
    for var, g in df_long.groupby("var"):
        arr = g["value"].to_numpy()
        mu = float(np.mean(arr))
        sd = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        half = z * sd / math.sqrt(len(arr)) if len(arr) > 1 else 0.0
        rows.append({"var": var, "mean": mu, "ci_low": mu - half, "ci_high": mu + half, "ci_halfwidth": half})
    return pd.DataFrame(rows)

# ------------- plots --------------------
def plot_errorbars(ci_df: pd.DataFrame, outfile: Path):
    order = [v for v in OUTPUT_VARS if v in ci_df["var"].tolist()]
    ci_df = ci_df.set_index("var").reindex(order).reset_index()
    x = np.arange(len(ci_df))
    y = ci_df["mean"].to_numpy()
    yerr = ci_df["ci_halfwidth"].to_numpy()

    fig = plt.figure(figsize=(7,4))
    ax = plt.subplot(111)
    ax.errorbar(x, y, yerr=yerr, fmt='o', capsize=4)
    ax.set_xticks(x); ax.set_xticklabels(ci_df["var"].tolist())
    ax.set_xlabel("Output variable"); ax.set_ylabel("Mean ± 95% CI")
    ax.set_title("Stability (Monte Carlo): Mean with 95% CI")
    plt.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=200)
    plt.close(fig)

def plot_boxplots(df_long: pd.DataFrame, outfile: Path):
    order = OUTPUT_VARS
    data = [df_long[df_long["var"]==v]["value"].to_numpy() for v in order]
    fig = plt.figure(figsize=(7,4))
    ax = plt.subplot(111)
    ax.boxplot(data, labels=order, showfliers=False)
    ax.set_title("Stability (Monte Carlo): Distributions")
    ax.set_xlabel("Output variable"); ax.set_ylabel("Value")
    plt.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=200)
    plt.close(fig)

# ------------- CLI ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="Path to your model .py")
    ap.add_argument("--samples", type=int, default=200, help="Monte Carlo samples (default 200)")
    ap.add_argument("--rel-range", type=float, default=0.10, help="Relative range ± (default 0.10)")
    ap.add_argument("--analysis-dir", type=str, default="./analysis_outputs", help="Folder to write CSV/figures")
    args = ap.parse_args()

    mod = import_model(args.model)
    analysis_dir = Path(args.analysis_dir)
    csv_path = monte_carlo(mod, args.samples, args.rel_range, analysis_dir)
    print(f"[OK] Monte Carlo samples CSV: {csv_path}")
    print(f"[OK] CI summary: {analysis_dir / 'monte_carlo_ci.csv'}")
    print(f"[OK] Figures: {analysis_dir / 'stability_errorbars.png'}, {analysis_dir / 'stability_boxplot.png'}")

if __name__ == "__main__":
    main()
