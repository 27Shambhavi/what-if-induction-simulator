#!/usr/bin/env python3
"""
induction_simulator.py (updated)

Usage examples:
  python induction_simulator.py
  python induction_simulator.py --input induction_input_converted.csv --num 6 --cleaning 2 --pm-threshold 0.8 --allow-expired

This script:
 - loads an induction input CSV (default tries: induction_input_converted.csv, induction_input.csv)
 - validates/coerces fields
 - computes composite scores (weights configurable inside)
 - selects top N respecting cleaning capacity / hard constraints
 - writes ranked_output.csv and selected_output.csv (or custom --out-dir)
"""

import argparse
import os
import sys
import math
import json
import pandas as pd
import numpy as np

# ---------------------------
# CONFIG: default weights
# ---------------------------
DEFAULT_WEIGHTS = {
    "pm_health": 0.30,
    "jobcard": 0.15,
    "fitness": 0.20,
    "branding": 0.10,
    "mileage": 0.10,
    "cleaning": 0.05,
    "stabling": 0.10
}

REQUIRED_COLS = {
    "trainset",
    "pm_failure_prob",
    "jobcard_open_frac",
    "minutes_to_latest_fitness_expiry",
    "branding_score",
    "mileage_need",
    "cleaning_required",
    "stabling_penalty",
    "manual_force_in",
    "manual_force_out",
}

# ---------------------------
# Helpers
# ---------------------------
def find_input_file(provided):
    # preference: provided path, induction_input_converted.csv, induction_input.csv
    candidates = []
    if provided:
        candidates.append(provided)
    candidates += ["induction_input_converted.csv", "induction_input.csv", "induction_input.csv"]
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None

def safe_read_csv(path):
    # try several encodings to handle BOM issues
    tries = ["utf-8", "utf-8-sig", "latin1"]
    last_err = None
    for enc in tries:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise last_err

def ensure_columns(df):
    # if mileage_need is missing but mileage_need_raw exists, normalize
    df_cols = set(df.columns.str.strip())
    if "mileage_need" not in df_cols and "mileage_need_raw" in df_cols:
        vals = pd.to_numeric(df["mileage_need_raw"].fillna(0)).astype(float)
        if vals.max() == vals.min():
            df["mileage_need"] = 0.0
        else:
            df["mileage_need"] = (vals - vals.min()) / (vals.max() - vals.min())
    # fill missing optional columns with defaults
    for col in ["manual_force_in", "manual_force_out", "cleaning_required"]:
        if col not in df.columns:
            df[col] = 0
    # coerce numeric columns
    numeric_cols = ["pm_failure_prob","jobcard_open_frac","minutes_to_latest_fitness_expiry",
                    "branding_score","mileage_need","cleaning_required","stabling_penalty",
                    "manual_force_in","manual_force_out"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].fillna(0), errors="coerce").fillna(0)
    # ensure trainset exists
    if "trainset" not in df.columns:
        raise ValueError("Input CSV must contain a 'trainset' column.")
    return df

def normalize_series(s):
    s = s.astype(float)
    if s.max() == s.min():
        return pd.Series([0.0]*len(s), index=s.index)
    return (s - s.min()) / (s.max() - s.min())

def compute_scores(df, weights):
    df = df.copy()
    # transform so higher is better
    df["score_pm_health"] = 1.0 - df["pm_failure_prob"].clip(0,1)
    df["score_jobcard"] = 1.0 - df["jobcard_open_frac"].clip(0,1)
    df["score_fitness_raw"] = df["minutes_to_latest_fitness_expiry"].clip(lower=-120)  # clamp
    df["score_branding"] = df["branding_score"].clip(0,1)
    df["score_mileage"] = df["mileage_need"].clip(0,1)
    df["score_cleaning"] = 1.0 - df["cleaning_required"].clip(0,1)
    # normalize fitness and stabling
    df["score_fitness"] = normalize_series(df["score_fitness_raw"])
    df["score_stabling"] = 1.0 - normalize_series(df["stabling_penalty"].astype(float))
    # composite
    df["composite_score"] = (
        weights["pm_health"]*df["score_pm_health"] +
        weights["jobcard"]*df["score_jobcard"] +
        weights["fitness"]*df["score_fitness"] +
        weights["branding"]*df["score_branding"] +
        weights["mileage"]*df["score_mileage"] +
        weights["cleaning"]*df["score_cleaning"] +
        weights["stabling"]*df["score_stabling"]
    )
    # contributions for explainability
    df["contrib_pm"] = (weights["pm_health"]*df["score_pm_health"]).round(4)
    df["contrib_jobcard"] = (weights["jobcard"]*df["score_jobcard"]).round(4)
    df["contrib_fitness"] = (weights["fitness"]*df["score_fitness"]).round(4)
    df["contrib_branding"] = (weights["branding"]*df["score_branding"]).round(4)
    df["contrib_mileage"] = (weights["mileage"]*df["score_mileage"]).round(4)
    df["contrib_cleaning"] = (weights["cleaning"]*df["score_cleaning"]).round(4)
    df["contrib_stabling"] = (weights["stabling"]*df["score_stabling"]).round(4)
    df["composite_score"] = df["composite_score"].round(6)
    return df

def detect_conflicts(df, pm_hard_threshold):
    alerts = []
    for _, r in df.iterrows():
        if r["minutes_to_latest_fitness_expiry"] < 0 and not bool(r.get("manual_force_in",0)):
            alerts.append({"trainset": r["trainset"], "alert": "FITNESS_EXPIRED"})
        if r["pm_failure_prob"] > pm_hard_threshold and not bool(r.get("manual_force_in",0)):
            alerts.append({"trainset": r["trainset"], "alert": "HIGH_PM_RISK"})
        if r["jobcard_open_frac"] > 0.7:
            alerts.append({"trainset": r["trainset"], "alert": "HIGH_JOBCARD_OPEN"})
    return alerts

def greedy_select(sorted_df, num_to_induct, cleaning_capacity, pm_hard_threshold, allow_expired):
    selected = []
    cleaning_used = 0
    for _, r in sorted_df.iterrows():
        if len(selected) >= num_to_induct:
            break
        # manual force-outs skip
        if bool(r.get("manual_force_out", False)):
            continue
        # forced in always include
        if bool(r.get("manual_force_in", False)):
            selected.append({"trainset": str(r["trainset"]), "reason":"FORCED_IN", "composite_score": float(r["composite_score"])})
            continue
        # hard constraints
        if (r["minutes_to_latest_fitness_expiry"] < 0) and (not allow_expired):
            continue
        if (r["pm_failure_prob"] > pm_hard_threshold):
            continue
        # cleaning capacity
        if int(r["cleaning_required"]) == 1 and cleaning_used >= cleaning_capacity:
            continue
        if int(r["cleaning_required"]) == 1:
            cleaning_used += 1
        selected.append({"trainset": str(r["trainset"]), "reason":"SELECTED", "composite_score": float(r["composite_score"])})
    return selected, cleaning_used

# ---------------------------
# Main CLI
# ---------------------------
def main():
    p = argparse.ArgumentParser(description="Induction What-If Simulator (updated)")
    p.add_argument("--input", "-i", help="input CSV path (default: autodetect)", default=None)
    p.add_argument("--out-dir", "-o", help="output directory", default=".")
    p.add_argument("--num", "-n", help="number to induct (default 6)", type=int, default=6)
    p.add_argument("--cleaning", "-c", help="cleaning capacity (default 2)", type=int, default=2)
    p.add_argument("--pm-threshold", help="PM hard threshold to exclude (default 0.8)", type=float, default=0.8)
    p.add_argument("--allow-expired", help="allow expired fitness (override)", action="store_true")
    args = p.parse_args()

    infile = find_input_file(args.input)
    if not infile:
        print("ERROR: No input CSV found. Place 'induction_input_converted.csv' or 'induction_input.csv' in the folder or pass --input path.", file=sys.stderr)
        sys.exit(2)

    print("Loading input:", infile)
    try:
        df = safe_read_csv(infile)
    except Exception as e:
        print("Failed to read input CSV:", str(e), file=sys.stderr)
        sys.exit(3)

    # normalize column names (strip spaces)
    df.columns = [c.strip() for c in df.columns]

    # ensure columns and types
    try:
        df = ensure_columns(df)
    except Exception as e:
        print("Input validation error:", str(e), file=sys.stderr)
        sys.exit(4)

    # compute scores
    df_scored = compute_scores(df, DEFAULT_WEIGHTS)
    # sort
    df_sorted = df_scored.sort_values("composite_score", ascending=False).reset_index(drop=True)

    # select using greedy algorithm
    selected, cleaning_used = greedy_select(df_sorted, args.num, args.cleaning, args.pm_threshold, args.allow_expired)

    # kpis
    expected_withdrawals = float(sum([df.loc[df["trainset"]==s["trainset"], "pm_failure_prob"].iloc[0] for s in selected])) if selected else 0.0
    total_stabling = float(sum([df.loc[df["trainset"]==s["trainset"], "stabling_penalty"].iloc[0] for s in selected])) if selected else 0.0
    branding_sum = float(sum([df.loc[df["trainset"]==s["trainset"], "branding_score"].iloc[0] for s in selected])) if selected else 0.0

    # detect conflicts on snapshot
    conflicts = detect_conflicts(df, args.pm_threshold)

    # outputs
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    ranked_out = os.path.join(out_dir, "ranked_output.csv")
    selected_out = os.path.join(out_dir, "selected_output.csv")

    # select columns to write for rank
    cols_to_write = ["trainset","composite_score","pm_failure_prob","jobcard_open_frac",
                     "minutes_to_latest_fitness_expiry","branding_score","mileage_need",
                     "cleaning_required","stabling_penalty",
                     "contrib_pm","contrib_jobcard","contrib_fitness","contrib_branding","contrib_mileage","contrib_cleaning","contrib_stabling"]
    available_cols = [c for c in cols_to_write if c in df_sorted.columns]
    df_sorted[available_cols].to_csv(ranked_out, index=False)
    pd.DataFrame(selected).to_csv(selected_out, index=False)

    # print summary
    print("\n--- SIMULATION SUMMARY ---")
    print("Input file:", infile)
    print("Number to induct requested:", args.num)
    print("Cleaning capacity:", args.cleaning)
    print("PM hard threshold:", args.pm_threshold)
    print("Allow expired fitness:", args.allow_expired)
    print("Selected count:", len(selected))
    print("Cleaning used (selected):", cleaning_used)
    print(f"Expected unscheduled withdrawals (sum of pm_prob for selected): {expected_withdrawals:.3f}")
    print(f"Total stabling cost (min) for selected: {total_stabling:.1f}")
    print(f"Branding exposure score (sum) for selected: {branding_sum:.3f}")
    print("Conflicts in snapshot (count):", len(conflicts))
    if conflicts:
        print(" Conflicts sample (up to 10):")
        for c in conflicts[:10]:
            print("  -", c["trainset"], ">>", c["alert"])
    print("\nOutputs written:")
    print(" Ranked:", ranked_out)
    print(" Selected:", selected_out)
    print("--------------------------\n")

if __name__ == "__main__":
    main()
