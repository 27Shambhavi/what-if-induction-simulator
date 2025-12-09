# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from dateutil import parser
from datetime import datetime, timezone

app = Flask(__name__)
CORS(app)  # allow requests from browser/dev tools

# -------------------------- DEFAULTS --------------------------
DEFAULT_WEIGHTS = {
    "pm_health": 0.30,
    "jobcard": 0.15,
    "fitness": 0.20,
    "branding": 0.10,
    "mileage": 0.10,
    "cleaning": 0.05,
    "stabling": 0.10,
}

REQUIRED_COLS = [
    "trainset", "pm_failure_prob", "jobcard_open_frac",
    "minutes_to_latest_fitness_expiry", "branding_score",
    "mileage_need", "cleaning_required", "stabling_penalty",
    "manual_force_in", "manual_force_out"
]

# -------------------------- helpers --------------------------
def normalize_series(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    if s.max() == s.min():
        return pd.Series([0.0] * len(s), index=s.index)
    return (s - s.min()) / (s.max() - s.min())

def compute_scores(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    df = df.copy()
    df["score_pm_health"] = 1 - df["pm_failure_prob"].clip(0,1)
    df["score_jobcard"] = 1 - df["jobcard_open_frac"].clip(0,1)
    df["score_fitness_raw"] = df["minutes_to_latest_fitness_expiry"].clip(lower=-120)
    df["score_branding"] = df["branding_score"].clip(0,1)
    df["score_mileage"] = df["mileage_need"].clip(0,1)
    df["score_cleaning"] = 1 - df["cleaning_required"].clip(0,1)
    df["score_fitness"] = normalize_series(df["score_fitness_raw"])
    df["score_stabling"] = 1 - normalize_series(df["stabling_penalty"].astype(float))
    df["composite_score"] = (
        weights["pm_health"]*df["score_pm_health"] +
        weights["jobcard"]*df["score_jobcard"] +
        weights["fitness"]*df["score_fitness"] +
        weights["branding"]*df["score_branding"] +
        weights["mileage"]*df["score_mileage"] +
        weights["cleaning"]*df["score_cleaning"] +
        weights["stabling"]*df["score_stabling"]
    )
    # round small floats
    df["composite_score"] = df["composite_score"].round(6)
    return df

def detect_conflicts(df: pd.DataFrame, pm_threshold: float):
    alerts = []
    for _, r in df.iterrows():
        if r["minutes_to_latest_fitness_expiry"] < 0 and not bool(r.get("manual_force_in",0)):
            alerts.append({"trainset": r["trainset"], "alert": "FITNESS_EXPIRED"})
        if r["pm_failure_prob"] > pm_threshold and not bool(r.get("manual_force_in",0)):
            alerts.append({"trainset": r["trainset"], "alert": "HIGH_PM_RISK"})
        if r["jobcard_open_frac"] > 0.7:
            alerts.append({"trainset": r["trainset"], "alert": "HIGH_JOBCARD_OPEN"})
    return alerts

def greedy_select(df_sorted: pd.DataFrame, num_to_induct: int, cleaning_capacity: int, pm_threshold: float, allow_expired: bool):
    selected = []
    cleaning_used = 0
    for _, r in df_sorted.iterrows():
        if len(selected) >= num_to_induct:
            break
        if bool(r.get("manual_force_out", False)):
            continue
        if bool(r.get("manual_force_in", False)):
            selected.append({"trainset": str(r["trainset"]), "reason":"FORCED_IN", "composite_score": float(r["composite_score"])})
            continue
        if (r["minutes_to_latest_fitness_expiry"] < 0) and (not allow_expired):
            continue
        if (r["pm_failure_prob"] > pm_threshold):
            continue
        if int(r["cleaning_required"]) == 1 and cleaning_used >= cleaning_capacity:
            continue
        if int(r["cleaning_required"]) == 1:
            cleaning_used += 1
        selected.append({"trainset": str(r["trainset"]), "reason":"SELECTED", "composite_score": float(r["composite_score"])})
    return selected, cleaning_used

# -------------------------- convert snapshot -> induction table --------------------------
def minutes_until(date_str):
    if not date_str:
        return -999
    dt = parser.parse(date_str)
    if dt.tzinfo is None:
        # assume local timezone Asia/Kolkata if naive (change if needed)
        dt = dt.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    return int((dt - now).total_seconds() / 60)

def convert_snapshot_to_table(snapshot: dict):
    """
    Accepts the JSON snapshot (branding_priorities, cleaning_slots, fitness_certificates,
    job_card_status, mileage, stabling_geometry) and returns a list of rows with the
    induction_input schema required by the simulator.
    """
    trains = {}
    mileage_map = {}

    # Branding -> assign branding_score (simple mapping)
    for b in snapshot.get("branding_priorities", []):
        tid = b.get("train_id")
        if not tid: 
            continue
        trains.setdefault(tid, {})
        lvl = b.get("priority_level", 3)
        score = 1.0 if lvl == 1 else (0.6 if lvl == 2 else 0.3)
        trains[tid]["branding_score"] = max(trains[tid].get("branding_score", 0.0), score)

    # Cleaning slots -> if a cleaning slot is scheduled within next 24h mark cleaning_required
    for c in snapshot.get("cleaning_slots", []):
        tid = c.get("train_id")
        if not tid:
            continue
        trains.setdefault(tid, {})
        start = c.get("slot_start")
        try:
            sdt = parser.parse(start)
            now = datetime.now(timezone.utc)
            # normalize tz
            if sdt.tzinfo is None:
                sdt = sdt.replace(tzinfo=timezone.utc)
            within_24h = 0 <= (sdt - now).total_seconds() <= 24*3600
            if within_24h:
                trains[tid]["cleaning_required"] = 1
            else:
                trains[tid].setdefault("cleaning_required", trains[tid].get("cleaning_required", 0))
        except Exception:
            trains[tid].setdefault("cleaning_required", trains[tid].get("cleaning_required", 0))

    # Stabling geometry -> approximate penalty
    for s in snapshot.get("stabling_geometry", []):
        tid = s.get("train_id")
        if not tid:
            continue
        trains.setdefault(tid, {})
        dist = float(s.get("distance_from_buffer_m", 0) or 0)
        track = float(s.get("track_no", 0) or 0)
        penalty = int(dist*3 + track)
        trains[tid]["stabling_penalty"] = penalty

    # Fitness certificates -> take minimum minutes to expiry among departments (worst-case)
    for f in snapshot.get("fitness_certificates", []):
        tid = f.get("train_id")
        if not tid:
            continue
        trains.setdefault(tid, {})
        mins = []
        for k in ("rolling_stock_validity","signalling_validity","telecom_validity"):
            v = f.get(k)
            if v:
                try:
                    mins.append(minutes_until(v))
                except Exception:
                    pass
        if mins:
            trains[tid]["minutes_to_latest_fitness_expiry"] = min(mins)

    # Jobcards -> compute open fraction per train
    job_counts = {}
    for j in snapshot.get("job_card_status", []):
        tid = j.get("train_id")
        if not tid:
            continue
        job_counts.setdefault(tid, {"total":0,"open":0})
        job_counts[tid]["total"] += 1
        status = str(j.get("status","")).lower()
        if status in ("pending","open"):
            job_counts[tid]["open"] += 1

    for tid, cnt in job_counts.items():
        trains.setdefault(tid, {})
        trains[tid]["jobcard_open_frac"] = cnt["open"]/max(1,cnt["total"])

    # Mileage -> delta_km -> normalize later
    for m in snapshot.get("mileage", []):
        tid = m.get("train_id")
        if not tid:
            continue
        mileage_map[tid] = float(m.get("delta_km", 0) or 0)

    # collect all train ids
    all_train_ids = set(list(trains.keys()) + list(mileage_map.keys()) + list(job_counts.keys()))
    # build rows
    rows = []
    for tid in sorted(all_train_ids):
        rec = trains.get(tid, {})
        delta_km = mileage_map.get(tid, 0)
        rows.append({
            "trainset": tid,
            "pm_failure_prob": 0.10,   # placeholder; replace with real PM model output
            "jobcard_open_frac": rec.get("jobcard_open_frac", 0.0),
            "minutes_to_latest_fitness_expiry": rec.get("minutes_to_latest_fitness_expiry", -999),
            "branding_score": rec.get("branding_score", 0.0),
            "mileage_need_raw": delta_km,
            "cleaning_required": rec.get("cleaning_required", 0),
            "stabling_penalty": rec.get("stabling_penalty", 10),
            "manual_force_in": 0,
            "manual_force_out": 0
        })
    # normalize mileage_need 0..1
    vals = [r["mileage_need_raw"] for r in rows]
    if vals:
        mn = min(vals)
        mx = max(vals)
        for r in rows:
            r["mileage_need"] = 0.0 if mx==mn else (r["mileage_need_raw"]-mn)/(mx-mn)
            r.pop("mileage_need_raw", None)
    return rows

# -------------------------- routes --------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "KMRL What-if Simulator API is Running 🚄"}), 200

@app.route("/simulate", methods=["POST"])
def simulate():
    """
    Accepts two payload styles:
    1) { "data": [ {trainset:.., pm_failure_prob:.., ...}, ... ], "params": { ... } }
       (i.e., pre-built feature table)
    2) { "snapshot": { ... } , "params": { ... } }
       (i.e., full Firebase-like snapshot JSON; server will convert to table)
    Return: selected, ranked, conflicts, kpis
    """
    try:
        payload = request.get_json(force=True)
        if payload is None:
            return jsonify({"error":"Empty JSON payload"}), 400

        params = payload.get("params", {})
        num_to_induct = int(params.get("num_to_induct", 6))
        cleaning_capacity = int(params.get("cleaning_capacity", 2))
        pm_threshold = float(params.get("pm_threshold", 0.8))
        allow_expired = bool(params.get("allow_expired", False))
        weights = params.get("weights", DEFAULT_WEIGHTS)

        if "data" in payload:
            df = pd.DataFrame(payload["data"])
        elif "snapshot" in payload:
            rows = convert_snapshot_to_table(payload["snapshot"])
            df = pd.DataFrame(rows)
        else:
            return jsonify({"error":"Payload must include either 'data' (table) or 'snapshot' (full JSON snapshot)."}), 400

        # ensure defaults for missing columns
        for c in REQUIRED_COLS:
            if c not in df.columns:
                # If expected numeric col missing, fill safe default
                df[c] = 0

        # coerce numeric types
        df["pm_failure_prob"] = pd.to_numeric(df["pm_failure_prob"], errors="coerce").fillna(0.0)
        df["jobcard_open_frac"] = pd.to_numeric(df["jobcard_open_frac"], errors="coerce").fillna(0.0)
        df["minutes_to_latest_fitness_expiry"] = pd.to_numeric(df["minutes_to_latest_fitness_expiry"], errors="coerce").fillna(-999)
        df["branding_score"] = pd.to_numeric(df["branding_score"], errors="coerce").fillna(0.0)
        df["mileage_need"] = pd.to_numeric(df["mileage_need"], errors="coerce").fillna(0.0)
        df["cleaning_required"] = pd.to_numeric(df["cleaning_required"], errors="coerce").fillna(0).astype(int)
        df["stabling_penalty"] = pd.to_numeric(df["stabling_penalty"], errors="coerce").fillna(0.0)
        df["manual_force_in"] = pd.to_numeric(df["manual_force_in"], errors="coerce").fillna(0).astype(int)
        df["manual_force_out"] = pd.to_numeric(df["manual_force_out"], errors="coerce").fillna(0).astype(int)

        df_scored = compute_scores(df, weights)
        df_sorted = df_scored.sort_values("composite_score", ascending=False).reset_index(drop=True)

        selected, cleaning_used = greedy_select(df_sorted, num_to_induct, cleaning_capacity, pm_threshold, allow_expired)
        conflicts = detect_conflicts(df, pm_threshold)

        expected_withdrawals = float(sum([df.loc[df["trainset"]==s["trainset"], "pm_failure_prob"].iloc[0] for s in selected])) if selected else 0.0
        total_stabling = float(sum([df.loc[df["trainset"]==s["trainset"], "stabling_penalty"].iloc[0] for s in selected])) if selected else 0.0
        branding_sum = float(sum([df.loc[df["trainset"]==s["trainset"], "branding_score"].iloc[0] for s in selected])) if selected else 0.0

        response = {
            "selected": selected,
            "ranked": df_sorted.to_dict(orient="records"),
            "conflicts": conflicts,
            "kpis": {
                "num_selected": len(selected),
                "cleaning_used": cleaning_used,
                "expected_withdrawals": expected_withdrawals,
                "total_stabling": total_stabling,
                "branding_sum": branding_sum
            }
        }
        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
