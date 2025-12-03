# convert_json_to_induction_csv.py
import json
from datetime import datetime, timezone
import pandas as pd
from dateutil import parser
import math

# load JSON snapshot (replace path)
with open("snapshot.json","r",encoding="utf-8") as f:
    data = json.load(f)

now = datetime.now(timezone.utc)  # use UTC or convert according to your timestamps

# helper maps
def minutes_until(date_str):
    if not date_str:
        return -999
    dt = parser.parse(date_str)
    # ensure timezone-aware
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int((dt - now).total_seconds() / 60)

# build per-train dictionaries
trains = {}
# init from branding priorities / stabling / fitness / cleaning / jobcards / mileage
for b in data.get("branding_priorities", []):
    tid = b["train_id"]
    trains.setdefault(tid, {})
    # simple branding_score: inverse of priority_level (scale to 0-1)
    # priority_level 1 -> 1.0, 2 -> 0.6, 3 -> 0.3 etc (tune as needed)
    if b.get("priority_level") == 1:
        score = 1.0
    elif b.get("priority_level") == 2:
        score = 0.6
    else:
        score = 0.3
    trains[tid]["branding_score"] = max(trains[tid].get("branding_score", 0), score)

for c in data.get("cleaning_slots", []):
    tid = c["train_id"]
    trains.setdefault(tid, {})
    # if a slot is scheduled within next 24h, mark cleaning_required
    start = parser.parse(c["slot_start"])
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    within_24h = (start - now).total_seconds() <= 24*3600 and (start - now).total_seconds() >= -3600
    if within_24h:
        trains[tid]["cleaning_required"] = 1
    else:
        trains[tid].setdefault("cleaning_required", trains[tid].get("cleaning_required", 0))

for s in data.get("stabling_geometry", []):
    tid = s["train_id"]
    trains.setdefault(tid, {})
    # stabling penalty proxy: distance_from_buffer_m * 3 + track_no
    dist = float(s.get("distance_from_buffer_m", 0))
    track = float(s.get("track_no", 0))
    penalty = int(dist*3 + track)
    trains[tid]["stabling_penalty"] = penalty

for f in data.get("fitness_certificates", []):
    tid = f["train_id"]
    trains.setdefault(tid, {})
    # take earliest expiry among the three departments (worst-case)
    vals = []
    for k in ["rolling_stock_validity","signalling_validity","telecom_validity"]:
        v = f.get(k)
        if v:
            vals.append(minutes_until(v))
    if vals:
        # minutes until nearest expiry (minimum)
        trains[tid]["minutes_to_latest_fitness_expiry"] = min(vals)

# jobcards: compute jobcard_open_frac
job_counts = {}
for j in data.get("job_card_status", []):
    tid = j["train_id"]
    job_counts.setdefault(tid, {"total":0,"open":0})
    job_counts[tid]["total"] += 1
    if j.get("status","").lower() in ("pending","open"):
        job_counts[tid]["open"] += 1

for tid,counts in job_counts.items():
    trains.setdefault(tid,{})
    trains[tid]["jobcard_open_frac"] = counts["open"]/max(1,counts["total"])

# mileage -> compute normalized mileage_need later (we will normalize across trains)
mileage_map = {}
for m in data.get("mileage", []):
    tid = m["train_id"]
    mileage_map[tid] = m.get("delta_km", 0)

# collect all train ids encountered
all_train_ids = set(trains.keys()) | set(mileage_map.keys()) | set(job_counts.keys())
# also include any train mentioned in branding_priorities
for b in data.get("branding_priorities", []):
    all_train_ids.add(b["train_id"])

rows = []
# placeholder pm_failure_prob: if you have PM model output use that; else set reasonable default
for tid in sorted(all_train_ids):
    rec = trains.get(tid, {})
    delta_km = mileage_map.get(tid, 0)
    # we'll set mileage_need = delta_km for now and normalize later
    rows.append({
        "trainset": tid,
        "pm_failure_prob": 0.10,  # replace from PM model
        "jobcard_open_frac": rec.get("jobcard_open_frac", 0.0),
        "minutes_to_latest_fitness_expiry": rec.get("minutes_to_latest_fitness_expiry", -999),
        "branding_score": rec.get("branding_score", 0.0),
        "mileage_need_raw": delta_km,
        "cleaning_required": rec.get("cleaning_required", 0),
        "stabling_penalty": rec.get("stabling_penalty", 10),
        "manual_force_in": 0,
        "manual_force_out": 0
    })

# normalize mileage_need to 0-1
vals = [r["mileage_need_raw"] for r in rows]
if max(vals) == min(vals):
    for r in rows:
        r["mileage_need"] = 0.0
else:
    mn, mx = min(vals), max(vals)
    for r in rows:
        r["mileage_need"] = (r["mileage_need_raw"] - mn) / (mx - mn) if mx>mn else 0.0
    for r in rows:
        del r["mileage_need_raw"]

# write CSV
df = pd.DataFrame(rows)
df = df[["trainset","pm_failure_prob","jobcard_open_frac","minutes_to_latest_fitness_expiry","branding_score","mileage_need","cleaning_required","stabling_penalty","manual_force_in","manual_force_out"]]
df.to_csv("induction_input_converted.csv", index=False)
print("Wrote induction_input_converted.csv with rows:", len(df))
