# -*- coding: utf-8 -*-
"""
CaCO3-assisted cement hydration dynamics (final public version)
- Inputs: time series of aqueous activities a_i (from GEMS), plus (optional) phase masses if you have them
- Outputs: time histories of phase amounts and diagnostics
- Only standard Python packages: pandas, numpy, pathlib

Core features:
1) Maps activities a_i -> ionic activity product Q -> saturation ratio Omega -> kinetic driving g(Omega).
2) CSH uses your total reaction (C/S=1.6 baseline). K_app is calibrated at a chosen step t* s.t. Omega_CSH(t*)=1.
3) AFt / AFm-HC / AFm-MC also computed from activities via configurable stoichiometry (can be tuned).
4) Two kinetic channels:
   - A (CSH nucleation/growth on CaCO3 surfaces):      R_A = A_eff * k_seed * g(Omega_CSH)
   - B (C3A system with competition + internal partition):
       R_AFt = A_eff * (1-H) * k_AFt * g(Omega_AFt)
       R_HC  = A_eff * H*(1-h_CO3) * k_AFm * g(Omega_HC) * F_imp
       R_MC  = A_eff * H*h_CO3    * k_AFm * g(Omega_MC) * F_imp

Author-facing notes:
- Edit the 'CONFIG' block to match species column names in activities.csv.
- If you want "time = GEMS node index", set time_from_csv=False (then Δt=1 for each row). If CSV提供 t 列，就设置 time_from_csv=True。
- This version avoids any SI provided by CSV; everything is computed from a_i.

"""

import numpy as np
import pandas as pd
from pathlib import Path

# ----------------------------- #
# 0) User configuration (EDIT)  #
# ----------------------------- #

CONFIG = {
    # === Files ===
    "activities_csv": "activities.csv",     # your Appendix-B-like table (see required columns below)
    "out_dir": "results",

    # === Which column is "time"? ===
    # If your CSV has a physical time column (e.g. 't_h' = hours), set:
    "time_from_csv": True,
    "time_col": "t",                        # change to your time column; if no time, set time_from_csv=False

    # === Temperature (K) (only used to tag outputs) ===
    "T_K": 298.15,

    # === dynamic coverage (optional) ---
    # base coverage plus adsorptive contributions; all clamped to [0, 0.99]
    "theta0": 0.0,          # base coverage (e.g., intrinsic organics)
    "K_cov_cit": 0.0,       # citrate-induced coverage strength
    "K_cov_Mg": 0.0,        # Mg-induced coverage strength
    "K_cov_I": 0.0,         # ionic-strength-induced coverage strength

    # === dynamic aggregation (optional) ---
    # if you want agg to grow with ionic strength, enable k_agg_I > 0
    "k_agg_I": 0.0,         # additional aggregation vs. ionic strength

    # === dynamic size function (optional) ---
    "size_model": "const",  # "const" | "power"
    "d50": 0.5,             # µm (example)
    "d_ref": 0.5,           # µm
    "m_size": 0.0,           # exponent in f_size=(d_ref/d50)^m

    # === CSH apparent equilibrium constant calibration ===
    # choose a row index (or time) where CSH is near-equilibrium -> Omega_CSH = 1
    # Option A: by row index
    "calibrate_by_index": 0,                # set to an integer index within CSV (0-based)
    # Option B (alternative): by time value -> choose the closest row (ignored if None)
    "calibrate_by_time_value": None,        # e.g. 24.0 (hours); set None to disable

    # === Kinetics parameters ===
    "k_seed": 5e-6,         # [mol m^-2 s^-1]  - CSH specific surface activity
    "k_AFt":  2e-6,         # [mol m^-2 s^-1]  - AFt area activity
    "k_AFm":  2e-6,         # [mol m^-2 s^-1]  - AFm area activity

    # === Driving function g(Omega) = max(Omega-1, eps)^n ===
    "g_n": 1.5,
    "g_eps": 1e-8,

    # === Competitive adsorption (B-channel) ===
    # Langmuir constants (dimensionless; activities already absorb units)
    "K_CO3": 1.0,
    "K_SO4": 1.0,
    "alpha_CO3": 1.0,   # weight for carbonate in H
    "beta_SO4":  1.0,   # weight for sulfate  in H
    # Mg penalty only in H: K_CO3_eff = K_CO3 / (1 + k_Mg * a_Mg)
    "k_Mg": 0.0,        # set >0 if you want Mg poisoning in H

    # === AFm internal partition h_CO3 ===
    "beta_delta": 4.0,  # steepness
    "delta_SI_star": 0.0,  # switch point

    # === Citrate inhibition for MCS (Al-bearing reactions) ===
    "K_cit": 0.0,       # set positive if citrate present; F_imp = 1/(1 + K_cit a_cit)

    # === Effective reacting area A_eff = phi_acc * A_geom (per 1 kg binder) ===
    "A_geom": 30.0,     # [m^2/kg] geometric/SSA
    # phi_acc = 1/(1 + a_agg*d_agg) * (1 - theta_cov) * f_size(d)
    "a_agg": 0.0, "d_agg": 0.0, "theta_cov": 0.0, "f_size": 1.0,

    # === Stoichiometry dictionaries for Q (activity product) ===
    # keys: CSV column names  (ensure your CSV uses the same labels)
    # ---------------- CSH total reaction (C/S=1.6 baseline) ----------------
    # 1.6 Ca2+ +  H4SiO4 +  1.2 OH-  ⇌  C–S–H1.6 + 2.8 H2O
    # -> Q_CSH = a_Ca^1.6 * a_H4SiO4^1.0 * a_OH^1.2
    "stoich_CSH": {"a_Ca": 1.6, "a_H4SiO4": 1.0, "a_OH": 1.2},

    # --------------- AFt proxy stoich (tunable; literature-based effective exponents) ---------------
    # Ettringite formation roughly involves Ca, Al(OH)4-, SO4^2-, OH-; you may tune exponents:
    "stoich_AFt": {"a_Ca": 3.0, "a_AlOH4": 2.0, "a_SO4": 3.0, "a_OH": 2.0},

    # --------------- AFm-HC proxy stoich (hemicarbonate) ---------------
    "stoich_HC": {"a_Ca": 4.0, "a_AlOH4": 2.0, "a_CO3": 0.5, "a_OH": 2.0},

    # --------------- AFm-MC proxy stoich (monocarbonate) ---------------
    "stoich_MC": {"a_Ca": 4.0, "a_AlOH4": 2.0, "a_CO3": 1.0, "a_OH": 2.0},

    # === Chemical equivalents for Ca per mole of each solid (for reporting) ===
    "nu_Ca_CSH": 1.6,   # mol Ca per mol CSH chosen representative
    "nu_Ca_AFt": 3.0,
    "nu_Ca_HC":  4.0,
    "nu_Ca_MC":  4.0,

    # === CSV required columns (rename here if different) ===
    # Time column only needed if time_from_csv=True
    "csv_cols": {
        "time": "t",          # if not present set time_from_csv=False
        "a_Ca": "a_Ca",
        "a_OH": "a_OH",
        "a_H4SiO4": "a_H4SiO4",
        "a_AlOH4": "a_AlOH4",
        "a_SO4": "a_SO4",
        "a_CO3": "a_CO3",
        "a_Mg": "a_Mg",       # optional; if not present, code will default to 0
        "a_cit": "a_cit"      # optional citrate
    }
}

# -------------------------------- #
# 1) Utilities and core functions  #
# -------------------------------- #

def safe_get(row, key, default=0.0):
    return float(row[key]) if key in row and pd.notna(row[key]) else float(default)

def geom_area_m2_perkg(cfg):
    phi_acc = 1.0 / (1.0 + cfg["a_agg"] * cfg["d_agg"])
    phi_acc *= (1.0 - cfg["theta_cov"])
    phi_acc *= cfg["f_size"]
    return cfg["A_geom"] * phi_acc

def omega_from_ai(row, stoich, K_app):
    """
    Build Q = product a_i^{nu_i}; Omega = Q / K_app; SI = log10(Omega)
    stoich: dict {csv_col: exponent}
    """
    Q = 1.0
    for col, power in stoich.items():
        if col not in row or pd.isna(row[col]):
            raise KeyError(f"Missing activity column '{col}' in CSV row.")
        Q *= float(row[col])**float(power)
    Omega = Q / float(K_app)
    SI = np.log10(max(Omega, 1e-300))
    return Omega, SI, Q

def g_of_omega(omega, n, eps):
    return max(omega - 1.0, eps)**n

def langmuir_gamma(KX, aX):
    return (KX * aX) / (1.0 + KX * aX)

def H_competition(row, cfg):
    # carbonate term (with Mg penalty if requested)
    aCO3 = safe_get(row, CONFIG["csv_cols"]["a_CO3"], 0.0)
    aSO4 = safe_get(row, CONFIG["csv_cols"]["a_SO4"], 0.0)
    aMg  = safe_get(row, CONFIG["csv_cols"]["a_Mg"],  0.0)

    KCO3_eff = cfg["K_CO3"] / (1.0 + cfg["k_Mg"] * aMg)
    Gc = langmuir_gamma(KCO3_eff, aCO3)
    Gs = langmuir_gamma(cfg["K_SO4"], aSO4)
    H = (1.0 + cfg["alpha_CO3"] * Gc) / (1.0 + cfg["alpha_CO3"] * Gc + cfg["beta_SO4"] * Gs)
    return np.clip(H, 0.0, 1.0), Gc, Gs

def h_CO3_partition(SI_MC, SI_HC, cfg):
    dSI = SI_MC - SI_HC
    return 1.0 / (1.0 + np.exp(-cfg["beta_delta"] * (dSI - cfg["delta_SI_star"])))

def F_imp_citrate(row, cfg):
    a_cit = safe_get(row, CONFIG["csv_cols"]["a_cit"], 0.0)
    if cfg["K_cit"] <= 0.0:
        return 1.0
    return 1.0 / (1.0 + cfg["K_cit"] * a_cit)

def pick_calibration_row(df, cfg):
    if cfg["calibrate_by_time_value"] is not None and cfg["time_from_csv"]:
        tcol = cfg["csv_cols"]["time"]
        idx = (df[tcol] - cfg["calibrate_by_time_value"]).abs().idxmin()
        return int(idx)
    return int(cfg["calibrate_by_index"])

def ionic_strength_from_row(row, cfg):
    # I ≈ 0.5*(a_Ca + a_Mg + a_SO4 + a_OH + a_CO3)
    get = lambda k: safe_get(row, CONFIG["csv_cols"][k], 0.0)
    return 0.5 * (get("a_Ca") + get("a_Mg") + get("a_SO4") + get("a_OH") + get("a_CO3"))

def dynamic_Aeff(row, cfg):
    """
    Compute A_eff at current step, accounting for aggregation, coverage, and size.
    All effects are optional and controlled by CONFIG.
    """
    # --- ionic strength surrogate (dimensionless here because we use activities) ---
    I = ionic_strength_from_row(row, cfg)

    # --- aggregation ---
    base_phi_agg = 1.0 / (1.0 + cfg["a_agg"] * cfg["d_agg"])
    phi_agg = base_phi_agg / (1.0 + cfg["k_agg_I"] * I)  # optional I-enhanced aggregation
    phi_agg = max(0.01, min(1.0, phi_agg))

    # --- coverage ---
    a_cit = safe_get(row, CONFIG["csv_cols"]["a_cit"], 0.0)
    a_Mg  = safe_get(row, CONFIG["csv_cols"]["a_Mg"],  0.0)
    theta = cfg["theta0"] \
          + cfg["K_cov_cit"] * a_cit/(1.0 + a_cit) \
          + cfg["K_cov_Mg"]  * a_Mg /(1.0 + a_Mg ) \
          + cfg["K_cov_I"]   * I    /(1.0 + I)
    theta = max(0.0, min(0.99, theta))  # avoid zero area

    # --- size model ---
    if cfg["size_model"] == "power":
        f_size = (cfg["d_ref"]/max(cfg["d50"], 1e-6))**cfg["m_size"]
    else:
        f_size = cfg["f_size"]

    A_eff = cfg["A_geom"] * phi_agg * (1.0 - theta) * f_size
    return A_eff


# ------------------------------ #
# 2) Main compute flow           #
# ------------------------------ #

def main(cfg):
    out_dir = Path(cfg["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(cfg["activities_csv"])
    if not csv_path.exists():
        raise FileNotFoundError(f"Cannot find input CSV: {csv_path.resolve()}")

    # Load activities
    df = pd.read_csv(csv_path)

    # Sanity-check columns; missing optional columns will be filled with zeros
    required = [cfg["csv_cols"][k] for k in ["a_Ca","a_OH","a_H4SiO4","a_AlOH4","a_SO4","a_CO3"]]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in {csv_path}")
    # optional
    for opt in ["a_Mg","a_cit"]:
        c = cfg["csv_cols"][opt]
        if c not in df.columns:
            df[c] = 0.0

    # build time array
    if cfg["time_from_csv"]:
        tcol = cfg["csv_cols"]["time"]
        if tcol not in df.columns:
            raise ValueError(f"time_from_csv=True but time column '{tcol}' not found")
        t = df[tcol].to_numpy(dtype=float)
    else:
        t = np.arange(len(df), dtype=float)  # n is hydration step; Δt = 1

    # Effective reacting area (per kg binder)
    for i in range(nrow):
        row = df.iloc[i].to_dict()
        # ...
        # --- dynamic effective area at this step ---
        A_eff = dynamic_Aeff(row, cfg)
        #  R_A, R_AFt, R_HC, R_MC

    # --- Calibrate K_app for CSH ---
    idx_star = pick_calibration_row(df, cfg)
    row_star = df.iloc[idx_star].to_dict()
    _, _, Q_star = omega_from_ai(row_star, cfg["stoich_CSH"], K_app=1.0)
    K_app_CSH = Q_star  # enforce Omega=1 at t*
    # (You can print it to keep record)
    print(f"[Calib] K_app_CSH set by row {idx_star}: {K_app_CSH:.5e}")

    # You may also set K_app for AF phases (or keep 1.0 as effective constant)
    K_app_AFt = 1.0
    K_app_HC  = 1.0
    K_app_MC  = 1.0

    # Storage
    nrow = len(df)
    out = {
        "t": t,
        "Omega_CSH": np.zeros(nrow), "SI_CSH": np.zeros(nrow),
        "Omega_AFt": np.zeros(nrow), "SI_AFt": np.zeros(nrow),
        "Omega_HC":  np.zeros(nrow), "SI_HC":  np.zeros(nrow),
        "Omega_MC":  np.zeros(nrow), "SI_MC":  np.zeros(nrow),
        "H": np.zeros(nrow), "h_CO3": np.zeros(nrow), "F_imp": np.ones(nrow),
        "R_A":  np.zeros(nrow),
        "R_AFt":np.zeros(nrow), "R_HC":np.zeros(nrow), "R_MC":np.zeros(nrow),
        "n_CSH": np.zeros(nrow), "n_AFt": np.zeros(nrow),
        "n_HC":  np.zeros(nrow), "n_MC":  np.zeros(nrow),
    }

    # integrate
    last_t = t[0]
    for i in range(nrow):
        row = df.iloc[i].to_dict()
        # dt
        dt = (t[i] - last_t) if i > 0 else (t[1]-t[0] if nrow>1 else 1.0)
        last_t = t[i]

        # ---- Omegas from a_i
        Om_CSH, SI_CSH, _ = omega_from_ai(row, cfg["stoich_CSH"], K_app=K_app_CSH)
        Om_AFt, SI_AFt, _ = omega_from_ai(row, cfg["stoich_AFt"], K_app=K_app_AFt)
        Om_HC,  SI_HC,  _ = omega_from_ai(row, cfg["stoich_HC"],  K_app=K_app_HC)
        Om_MC,  SI_MC,  _ = omega_from_ai(row, cfg["stoich_MC"],  K_app=K_app_MC)

        out["Omega_CSH"][i], out["SI_CSH"][i] = Om_CSH, SI_CSH
        out["Omega_AFt"][i], out["SI_AFt"][i] = Om_AFt, SI_AFt
        out["Omega_HC"][i],  out["SI_HC"][i]  = Om_HC,  SI_HC
        out["Omega_MC"][i],  out["SI_MC"][i]  = Om_MC,  SI_MC

        # ---- Channel A
        rA = cfg["k_seed"] * g_of_omega(Om_CSH, cfg["g_n"], cfg["g_eps"])
        R_A = A_eff * rA

        # ---- Channel B helpers
        H, Gc, Gs = H_competition(row, cfg)
        h = h_CO3_partition(SI_MC, SI_HC, cfg)
        Fimp = F_imp_citrate(row, cfg)

        # ---- Channel B rates
        rAFt = cfg["k_AFt"] * g_of_omega(Om_AFt, cfg["g_n"], cfg["g_eps"])
        rHC  = cfg["k_AFm"] * g_of_omega(Om_HC,  cfg["g_n"], cfg["g_eps"])
        rMC  = cfg["k_AFm"] * g_of_omega(Om_MC,  cfg["g_n"], cfg["g_eps"])

        R_AFt = A_eff * (1.0 - H) * rAFt
        R_HC  = A_eff * H * (1.0 - h) * rHC * Fimp
        R_MC  = A_eff * H * h        * rMC * Fimp

        # ---- Store rates
        out["R_A"][i]   = R_A
        out["R_AFt"][i] = R_AFt
        out["R_HC"][i]  = R_HC
        out["R_MC"][i]  = R_MC
        out["H"][i]     = H
        out["h_CO3"][i] = h
        out["F_imp"][i] = Fimp

        # ---- Explicit Euler integration (moles per kg binder)
        out["n_CSH"][i] = (out["n_CSH"][i-1] + R_A   * dt) if i>0 else R_A   * dt
        out["n_AFt"][i] = (out["n_AFt"][i-1] + R_AFt * dt) if i>0 else R_AFt * dt
        out["n_HC"][i]  = (out["n_HC"][i-1]  + R_HC  * dt) if i>0 else R_HC  * dt
        out["n_MC"][i]  = (out["n_MC"][i-1]  + R_MC  * dt) if i>0 else R_MC  * dt

    # Save results
    out_df = pd.DataFrame(out)
    out_df.to_csv(out_dir / "time_series_results.csv", index=False)
    print(f"[OK] Results written to {out_dir / 'time_series_results.csv'}")

    # Also dump a small README of the calibration
    with open(out_dir / "Kapp_calibration.txt", "w", encoding="utf-8") as f:
        f.write(f"K_app_CSH calibrated at row {idx_star} (time={t[idx_star] if len(t)>idx_star else 'NA'}): {K_app_CSH:.6e}\n")
        f.write("AF phases use K_app=1.0 by default (effective constants). Tune 'stoich_*' and k_* if needed.\n")

if __name__ == "__main__":
    main(CONFIG)
