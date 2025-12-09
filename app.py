# app.py
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from dateutil import parser
from datetime import datetime, timezone

app = Flask(__name__)

# Default weights (you can tune)
DEFAULT_WEIGHTS = {
    "pm_health": 0.30,
    "jobcard": 0.15,
    "fitness": 0.20,
    "branding": 0.10,
    "mileage": 0.10,
    "cleaning": 0.05,
    "stabling": 0.10,
}

# ---------- Helper utilities ----------
def normalize_series(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    if s.max() == s.min():
        return pd.Series([0.0] * len(s), index=s.index)
    return (s - s.min()) / (s.max() - s.min())

def compute_scores(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    df = df.copy()
    df["score_pm_health"] = 1.0 - df["pm_failure_prob"].clip(0,1)
    df["score_jobcard"] = 1.0 - df["jobcard_open_frac"].clip(0,1)
    df["score_fitness_raw"] = df["minutes_to_latest_fitness_expiry"].clip(lower=-120)
    df["score_branding"] = df["branding_score"].clip(0,1)
    df["score_mileage"] = df["mileage_need"].clip(0,1)
    df["score_cleaning"] = 1.0 - df["cleaning_required"].clip(0,1)

    df["score_fitness"] = normalize_series(df["score_fitness_raw"])
    df["score_stabling"] = 1.0 - normalize_series(df["stabling_penalty"].astype(float))

    df["composite_score"] = (
        weights["pm_health"] * df["score_pm_health"]
        + weights["jobcard"] * df["score_jobcard"]
        + weights["fitness"] * df["score_fitness"]
        + weights["branding"] * df["score_branding"]
        + weights["mileage"] * df["score_mileage"]
        + weights["cleaning"] * df["score_cleaning"]
        + weights["stabling"] * df["score_stabling"]
    )

    # round for nicer JSON
    df["composite_score"] = df["composite_score"].round(6)
    return df

def detect_conflicts(df: pd.DataFrame, pm_threshold: float):
    alerts = []
    for _, r in df.iterrows():
        if r["minutes_to_latest_fitness_expiry"] < 0 and not bool(r.get("manual_force_in", 0)):
            alerts.append({"trainset": r["trainset"], "alert": "FITNESS_EXPIRED"})
        if r["pm_failure_prob"] > pm_threshold and not bool(r.get("manual_force_in", 0)):
            alerts.append({"trainset": r["trainset"], "alert": "HIGH_PM_RISK"})
        if r["jobcard_open_frac"] > 0.7:
            alerts.append({"trainset": r["trainset"], "alert": "HIGH_JOBCARD_OPEN"})
    return alerts

def greedy_select(df_sorted: pd.DataFrame, num_to_induct: int, cleaning_capacity: int,
                  pm_threshold: float, allow_expired: bool):
    selected = []
    cleaning_used = 0
    for _, r in df_sorted.iterrows():
        if len(selected) >= num_to_induct:
            break
        # manual force-out skip
        if bool(r.get("manual_force_out", False)):
            continue
        # manual force-in include immediately
        if bool(r.get("manual_force_in", False)):
            selected.append({"trainset": str(r["trainset"]), "reason":"FORCED_IN", "composite_score": float(r["composite_score"])})
            continue
        # hard constraints
        if (r["minutes_to_latest_fitness_expiry"] < 0) and (not allow_expired):
            continue
        if (r["pm_failure_prob"] > pm_threshold):
            continue
        # cleaning capacity constraint
        if int(r.get("cleaning_required",0)) == 1 and cleaning_used >= cleaning_capacity:
            continue
        if int(r.get("cleaning_required",0)) == 1:
            cleaning_used += 1
        selected.append({"trainset": str(r["trainset"]), "reason":"SELECTED", "composite_score": float(r["composite_score"])})
    return selected, cleaning_used

# Convert the JSON snapshot structure into induction table rows the simulator expects
def snapshot_to_table(snapshot: dict, now_utc: datetime = None) -> pd.DataFrame:
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)

    trains = {}  # aggregator per train

    # branding_priorities -> branding_score (simple mapping of priority to score)
    for b in snapshot.get("branding_priorities", []):
        tid = b.get("train_id")
        trains.setdefault(tid, {})
        lvl = b.get("priority_level", 3)
        if lvl == 1:
            score = 1.0
        elif lvl == 2:
            score = 0.6
        else:
            score = 0.3
        trains[tid]["branding_score"] = max(trains[tid].get("branding_score", 0.0), score)

    # cleaning_slots -> cleaning_required if slot starts within next 24h
    for c in snapshot.get("cleaning_slots", []):
        tid = c.get("train_id")
        trains.setdefault(tid, {})
        start = c.get("slot_start")
        try:
            dt = parser.parse(start)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            within_24h = 0 <= (dt - now_utc).total_seconds() <= 24*3600
        except Exception:
            within_24h = False
        trains[tid]["cleaning_required"] = int(within_24h or trains[tid].get("cleaning_required", 0))

    # stabling_geometry -> stabling_penalty proxy
    for s in snapshot.get("stabling_geometry", []):
        tid = s.get("train_id")
        trains.setdefault(tid, {})
        dist = float(s.get("distance_from_buffer_m", 0.0) or 0.0)
        track = float(s.get("track_no", 0) or 0)
        penalty = int(dist*3 + track)  # simple proxy
        trains[tid]["stabling_penalty"] = penalty

    # fitness_certificates -> minutes until the earliest expiry among departments (worst-case)
    for f in snapshot.get("fitness_certificates", []):
        tid = f.get("train_id")
        trains.setdefault(tid, {})
        mins = []
        for k in ("rolling_stock_validity","signalling_validity","telecom_validity"):
            v = f.get(k)
            if v:
                try:
                    dt = parser.parse(v)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    mins.append(int((dt - now_utc).total_seconds()/60))
                except Exception:
                    pass
        if mins:
            # worst-case nearest expiry
            trains[tid]["minutes_to_latest_fitness_expiry"] = min(mins)

    # job_card_status -> jobcard_open_frac per train
    job_counts = {}
    for j in snapshot.get("job_card_status", []):
        tid = j.get("train_id")
        job_counts.setdefault(tid, {"total":0,"open":0})
        job_counts[tid]["total"] += 1
        if str(j.get("status","")).lower() in ("pending","open"):
            job_counts[tid]["open"] += 1
    for tid,counts in job_counts.items():
        trains.setdefault(tid,{})
        trains[tid]["jobcard_open_frac"] = counts["open"]/max(1,counts["total"])

    # mileage -> gather delta_km (normalize later)
    mileage_map = {}
    for m in snapshot.get("mileage", []):
        tid = m.get("train_id")
        mileage_map[tid] = float(m.get("delta_km", 0) or 0.0)

    # Collect all train ids
    all_train_ids = set(trains.keys()) | set(mileage_map.keys())
    # build rows
    rows = []
    for tid in sorted(all_train_ids):
        rec = trains.get(tid, {})
        rows.append({
            "trainset": tid,
            "pm_failure_prob": float(0.10),  # placeholder; ideally supply PM model output here
            "jobcard_open_frac": float(rec.get("jobcard_open_frac", 0.0)),
            "minutes_to_latest_fitness_expiry": int(rec.get("minutes_to_latest_fitness_expiry", -999)),
            "branding_score": float(rec.get("branding_score", 0.0)),
            "mileage_need_raw": float(mileage_map.get(tid, 0.0)),
            "cleaning_required": int(rec.get("cleaning_required", 0)),
            "stabling_penalty": int(rec.get("stabling_penalty", 10)),
            "manual_force_in": 0,
            "manual_force_out": 0
        })

    # normalize mileage_need
    vals = [r["mileage_need_raw"] for r in rows]
    mn = min(vals) if vals else 0.0
    mx = max(vals) if vals else 0.0
    for r in rows:
        if mx == mn:
            r["mileage_need"] = 0.0
        else:
            r["mileage_need"] = (r["mileage_need_raw"] - mn)/(mx-mn) if mx>mn else 0.0
        r.pop("mileage_need_raw", None)

    df = pd.DataFrame(rows)
    # ensure defaults
    for col in ["pm_failure_prob","jobcard_open_frac","minutes_to_latest_fitness_expiry","branding_score","mileage_need","cleaning_required","stabling_penalty","manual_force_in","manual_force_out"]:
        if col not in df.columns:
            df[col] = 0
    return df

# ---------- Routes ----------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "KMRL What-if Simulator API is Running 🚄"})

@app.route("/simulate", methods=["POST"])
def simulate():
    """
    Accepts either:
    - JSON snapshot in the same shape as your example (top-level keys branding_priorities, cleaning_slots, etc.) OR
    - an object with key "data" = table rows (list of dict) matching induction_input schema.
    Body JSON example 1 (snapshot):
      { <your snapshot JSON> , "num_to_induct":6, "cleaning_capacity":2, "pm_threshold":0.8 }
    Body JSON example 2 (direct rows):
      { "data": [ {trainset:..., pm_failure_prob:...}, ... ], "num_to_induct":6, ... }
    """
    try:
        payload = request.get_json(force=True)
    except Exception as e:
        return jsonify({"error":"Invalid JSON body", "details": str(e)}), 400

    # parameters
    num_to_induct = int(payload.get("num_to_induct", 6))
    cleaning_capacity = int(payload.get("cleaning_capacity", 2))
    pm_threshold = float(payload.get("pm_threshold", 0.8))
    allow_expired = bool(payload.get("allow_expired", False))
    weights = payload.get("weights", DEFAULT_WEIGHTS)

    # accept either direct table or snapshot
    if "data" in payload:
        df = pd.DataFrame(payload["data"])
    else:
        try:
            df = snapshot_to_table(payload)
        except Exception as e:
            return jsonify({"error":"Failed to convert snapshot to table", "details": str(e)}), 400

    # basic validation: coerce columns
    expected_cols = ["trainset","pm_failure_prob","jobcard_open_frac","minutes_to_latest_fitness_expiry","branding_score","mileage_need","cleaning_required","stabling_penalty","manual_force_in","manual_force_out"]
    for c in expected_cols:
        if c not in df.columns:
            df[c] = 0
    # ensure numeric types
    numeric_cols = [c for c in expected_cols if c!="trainset"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c].fillna(0), errors="coerce").fillna(0)

    # score and select
    df_scored = compute_scores(df, weights)
    df_sorted = df_scored.sort_values("composite_score", ascending=False).reset_index(drop=True)

    selected, cleaning_used = greedy_select(df_sorted, num_to_induct, cleaning_capacity, pm_threshold, allow_expired)
    conflicts = detect_conflicts(df, pm_threshold)

    # KPIs
    expected_withdrawals = float(sum([float(df.loc[df["trainset"]==s["trainset"], "pm_failure_prob"].iloc[0]) for s in selected])) if selected else 0.0
    total_stabling = float(sum([float(df.loc[df["trainset"]==s["trainset"], "stabling_penalty"].iloc[0]) for s in selected])) if selected else 0.0
    branding_sum = float(sum([float(df.loc[df["trainset"]==s["trainset"], "branding_score"].iloc[0]) for s in selected])) if selected else 0.0

    # build response
    resp = {
        "selected": selected,
        "ranked": df_sorted.to_dict(orient="records"),
        "conflicts": conflicts,
        "kpis": {
            "num_to_induct": num_to_induct,
            "cleaning_used": cleaning_used,
            "expected_unscheduled_withdrawals": round(expected_withdrawals,4),
            "total_stabling_cost_min": round(total_stabling,2),
            "branding_exposure_sum": round(branding_sum,4),
        }
    }
    return jsonify(resp), 200

if __name__ == "__main__":
    # for local dev; in production use gunicorn
    app.run(host="0.0.0.0", port=5000, debug=False)
