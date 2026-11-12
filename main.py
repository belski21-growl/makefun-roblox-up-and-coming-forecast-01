#!/usr/bin/env python3
"""Roblox Up-and-Coming top-10 forecaster (make.fun roblox-up-and-coming-forecast-01).

Reads /input/history.ndjson + /input/task.json, writes /output/forecast.json:
one static forecast scored against every hourly read of the 72h window.

Model (fit from the kit history; see build-log.md for the evidence):
- The chart churns fast (top-10 overlap 6.9/10 at +24h, 5.0 at +48h, 3.6 at +72h).
  Rank velocity itself does NOT persist (corr 0.07), but log playing-growth does:
  past-24h log playing growth predicts next-24h rank change with slope -14.6
  (corr -0.33); the signal is gone by day 2 (slope +2.4), so velocity decays
  with a ~24h time constant.
- Each game gets a projected rank path r(t) = r0 + D * (1 - exp(-t/TAU)),
  D = -GROWTH_SLOPE * (recent daily log playing growth, centred on the median),
  and a Gaussian spread sigma(t) = sqrt(SIGMA0^2 + SIGMA_DAY^2 * t/24) around it.
  The forecast is the time-average of that Gaussian mass over the window hours.
- Per place, named mass is scaled to (1 - unnamed); the unnamed share is spread
  over every game the read's chart holds that was not named there (the scorer's rule).
- Games that would age out (published >28 days ago) are faded using details.created
  when the history carries details (the house recorder's reads do; the archive's don't).

Standard library only, no network, fully deterministic (no randomness at all).
"""
import json
import math
import os
from datetime import datetime, timezone

PARAMS = {
    "half_life_h": 24.0,     # recency weighting for the growth fit
    "growth_window_h": 48.0,  # only snapshots this fresh feed the growth fit
    "growth_slope": 20.0,    # ranks per unit daily log playing growth
    "disp_cap": 25.0,        # |D| clamp, ranks
    "tau_h": 36.0,           # velocity decay time constant, hours
    "sigma0": 2.0,           # rank spread at t=0
    "sigma_day": 8.0,        # diffusive spread growth, ranks per sqrt(day)
    "unnamed": [0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20],
    "name_top": 60,          # max named games per place (rule allows 100)
    "r0_smooth": 0.7,        # weight of last rank vs previous snapshot rank
    "max_age_days": 28.0,    # chart only lists games published in the last 28 days
}


def _parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _wlinfit(xs, ys, ws):
    """Weighted least squares y ~ a + b*x. Returns (a, b)."""
    sw = sum(ws)
    if sw <= 0:
        return None
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return (my, 0.0)
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    return (my - b * mx, b)


def load_history(input_dir):
    snaps = []
    path = os.path.join(input_dir, "history.ndjson")
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if not d.get("ok", True):
                continue
            rows = d.get("up_and_coming") or []
            if not rows:
                continue
            try:
                t = _parse_time(d["at"])
            except (KeyError, ValueError):
                continue
            chart = {}
            for r in rows:
                try:
                    uid = int(r["universe_id"])
                    rank = int(r["rank"])
                except (KeyError, TypeError, ValueError):
                    continue
                if uid not in chart or rank < chart[uid][0]:
                    playing = r.get("playing")
                    chart[uid] = (rank, playing if isinstance(playing, (int, float)) else None)
            details = d.get("details") or {}
            snaps.append({"t": t, "chart": chart, "details": details})
    snaps.sort(key=lambda s: s["t"])
    return snaps


def load_task(input_dir):
    task = {"places": 10, "output": "forecast.json", "window": {"slots": 72, "slot_minutes": 60}}
    path = os.path.join(input_dir, "task.json")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            task.update(json.load(fh))
    except (OSError, ValueError):
        pass
    return task


def build_forecast(input_dir, params=None):
    p = dict(PARAMS)
    if params:
        p.update(params)
    task = load_task(input_dir)
    places = int(task.get("places") or 10)
    window = task.get("window") or {}
    n_slots = int(window.get("slots") or 72)
    slot_min = int(window.get("slot_minutes") or 60)
    snaps = load_history(input_dir)
    if not snaps:
        return {"probabilities": [], "top10": list(range(1, places + 1))}

    t_end = snaps[-1]["t"]
    n_chart = max(len(s["chart"]) for s in snaps[-2:]) if len(snaps) >= 2 else len(snaps[-1]["chart"])
    if n_chart <= 0:
        n_chart = 97

    # --- per-game series: (hours before t_end, rank, log playing) ---
    series = {}
    created = {}
    for s in snaps:
        age_h = (s["t"] - t_end).total_seconds() / 3600.0  # <= 0
        for uid, (rank, playing) in s["chart"].items():
            series.setdefault(uid, []).append((age_h, rank, playing))
        for uid, det in (s["details"] or {}).items():
            try:
                uid_i = int(uid)
            except (TypeError, ValueError):
                continue
            c = (det or {}).get("created")
            if c and uid_i not in created:
                try:
                    created[uid_i] = _parse_time(c)
                except ValueError:
                    pass

    half_life = p["half_life_h"]
    gw = p["growth_window_h"]
    growths = {}
    r0s = {}
    for uid, obs in series.items():
        obs.sort()
        # r0: last rank, lightly smoothed with the previous snapshot's rank
        r_last = obs[-1][1]
        r_prev = obs[-2][1] if len(obs) >= 2 else r_last
        r0 = p["r0_smooth"] * r_last + (1.0 - p["r0_smooth"]) * r_prev
        r0s[uid] = r0
        # log playing growth slope (per hour) over the recent window
        xs, ys, ws = [], [], []
        for age_h, rank, playing in obs:
            if age_h < -gw or playing is None or playing <= 0:
                continue
            xs.append(age_h)
            ys.append(math.log(playing))
            ws.append(0.5 ** (-age_h / half_life))
        if len(xs) >= 2:
            fit = _wlinfit(xs, ys, ws)
            growths[uid] = fit[1] if fit else 0.0
        else:
            growths[uid] = None

    med_growth = 0.0
    gs = sorted(g for g in growths.values() if g is not None)
    if gs:
        med_growth = gs[len(gs) // 2]

    # window start: from task.json if present, else t_end
    wfrom = None
    try:
        wfrom = _parse_time(task["window"]["from"])
    except (KeyError, TypeError, ValueError):
        wfrom = t_end
    slot_hours = slot_min / 60.0

    # --- time-averaged Gaussian mass per game per place ---
    # mass[place][uid] = mean over slots of P(game at place)
    mass = {k: {} for k in range(1, places + 1)}
    tau = p["tau_h"]
    cap = p["disp_cap"]
    slope = p["growth_slope"]
    for uid, r0 in r0s.items():
        g = growths.get(uid)
        if g is None:
            g = med_growth
        disp = -slope * (g - med_growth) * 24.0  # total rank displacement, ranks
        disp = max(-cap, min(cap, disp))
        # age-out: game leaves the chart once it is max_age_days old
        expire_h = None
        if uid in created:
            age_at_end_h = (t_end - created[uid]).total_seconds() / 3600.0
            expire_h = p["max_age_days"] * 24.0 - age_at_end_h  # hours after t_end
        for h in range(n_slots):
            # slot h starts at wfrom + h*slot_hours; centre the hour
            t_h = (wfrom - t_end).total_seconds() / 3600.0 + (h + 0.5) * slot_hours
            if t_h < 0:
                t_h = 0.0
            if expire_h is not None and t_h >= expire_h:
                continue  # aged out of the chart by this hour
            r_t = r0 + disp * (1.0 - math.exp(-t_h / tau)) if tau > 0 else r0 + disp
            sig = math.sqrt(p["sigma0"] ** 2 + p["sigma_day"] ** 2 * (t_h / 24.0))
            if sig <= 0:
                sig = 1e-9
            r_t = max(1.0, min(float(n_chart) + 20.0, r_t))
            for k in range(1, places + 1):
                m = _norm_cdf((k + 0.5 - r_t) / sig) - _norm_cdf((k - 0.5 - r_t) / sig)
                if m > 1e-12:
                    mass[k][uid] = mass[k].get(uid, 0.0) + m / n_slots

    # --- per-place normalisation with unnamed reserve ---
    unnamed = p["unnamed"]
    probabilities = []
    for k in range(1, places + 1):
        u = unnamed[k - 1] if k - 1 < len(unnamed) else unnamed[-1]
        items = sorted(mass[k].items(), key=lambda kv: (-kv[1], kv[0]))[: p["name_top"]]
        total = sum(v for _, v in items)
        if total <= 0:
            continue
        scale = (1.0 - u) / total
        for uid, v in items:
            pv = v * scale
            if pv >= 0.000001:
                probabilities.append({"universe_id": int(uid), "place": k, "p": round(pv, 6)})
    probabilities.sort(key=lambda r: (r["place"], -r["p"], r["universe_id"]))

    # --- top10: rank games by total named probability, order by most likely place ---
    totals = {}
    best_place = {}
    for row in probabilities:
        totals[row["universe_id"]] = totals.get(row["universe_id"], 0.0) + row["p"]
        if row["universe_id"] not in best_place or row["p"] > best_place[row["universe_id"]][0]:
            best_place[row["universe_id"]] = (row["p"], row["place"])
    ranked = sorted(totals.items(), key=lambda kv: (-kv[1], best_place[kv[0]][1], kv[0]))
    top10 = []
    for uid, _ in ranked[:places]:
        top10.append(uid)
    # order top10 by most likely place, then total mass
    top10.sort(key=lambda uid: (best_place[uid][1], -totals[uid], uid))
    # fill from the last chart if fewer than `places` candidates (should not happen)
    if len(top10) < places:
        last_chart = sorted(snaps[-1]["chart"].items(), key=lambda kv: kv[1][0])
        for uid, _ in last_chart:
            if uid not in top10:
                top10.append(uid)
            if len(top10) >= places:
                break
    top10 = top10[:places]
    return {"probabilities": probabilities, "top10": [int(x) for x in top10]}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    task = load_task(input_dir)
    fc = build_forecast(input_dir)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, task.get("output") or "forecast.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(fc, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
