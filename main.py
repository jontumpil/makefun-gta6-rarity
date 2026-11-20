"""gta6-trophy-rarity-forecast-01 entry: a count-consistent refresh forecaster.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0) in python:3.13-slim,
standard library only, no network. It reads every <id>.in in the input directory (the grader
runs one case at a time) and writes <id>.out per case: {"trophies": [{"id": n, "q": [q05, q25,
q50, q75, q95]}, ...]}, five quantiles of each trophy's Exophase earn rate (percent) at the
case's target. Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR stand
in for /input and /output.

The model, in one paragraph: the house's reads are not the process — Exophase refreshes its
counts on its own schedule, so consecutive identical reads are one *refresh state*, and only
the states carry information. In state space the counts are what move smoothly: players
tracked P and, per trophy, earners E = rate * P / 100 (the page's trophies_earned is their
sum). We fit log-count growth against x = ln(1 + hours since the first state) by recency-
weighted least squares — one slope for P, one for total earners, one per trophy shrunk toward
the total — extrapolate each count to the target, rescale the per-trophy earners so their sum
matches the extrapolated total (count consistency), and divide by the extrapolated P. A rate
therefore falls exactly when its earners grow slower than the player base (dilution) and rises
when they grow faster (progression), and no forecast can imply more earners than players.
With a single state nothing is identifiable, so the priors take over: counts keep growing at
launch-like slopes, and the median takes a dilution drift, because a page that has not
refreshed in hours is due one and a launch refresh dilutes. The spread is the baseline's
logit random walk widened by the fit's residual volatility over the extrapolation distance
and by refresh uncertainty, and it leans toward the side the median is drifting away from.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module shipped beside
# main.py would import. (This entry is a single file; the line costs nothing.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# --- Model constants (fixed in advance of any answer; see the docstring) ---
PRIOR_PLAYER_SLOPE = 1.10   # ln-players growth per unit x when no state pair identifies it
PRIOR_EARN_SLOPE = 1.30     # ln-earners growth per unit x, same role
SHRINK_K = 3.0              # per-trophy slopes shrink toward the total-earners slope
SLOPE_MIN, SLOPE_MAX = -0.3, 3.5
SINGLE_STATE_DRIFT = 0.8    # logit dilution drift per sqrt(horizon day), one-state cases
RW_SIGMA = 0.5              # the baseline's logit random-walk spread per sqrt day
DEFAULT_VOL = 0.30          # per-trophy ln-earner residual volatility when unfitted


def clamp_pct(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_pct(pct)
    return math.log(p / (100 - p))


def inv_logit(x: float) -> float:
    if x >= 0:
        return 100 / (1 + math.exp(-x))
    e = math.exp(x)
    return 100 * e / (1 + e)


def round4(x: float) -> float | int:
    """x rounded to 4 decimals, half up, as JavaScript's Math.round(x * 10000) / 10000 rounds it.

    A whole number comes back as an int, so json.dumps writes 100 where JavaScript writes 100.
    """
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)


def refresh_states(case: dict) -> list[dict]:
    """Compress the case's reads to Exophase refresh states.

    A new state starts when the players-tracked count or any trophy's rate changes; reads
    that repeat the page exactly only tell us the state lasted, which the state's first and
    last read times record.
    """
    states: list[dict] = []
    for read in case["history"]:
        key = (read["players_tracked"], tuple(read["rates"]))
        if states and states[-1]["key"] == key:
            states[-1]["last_at"] = parse_time(read["at"])
            continue
        states.append({
            "key": key,
            "first_at": parse_time(read["at"]),
            "last_at": parse_time(read["at"]),
            "players": read["players_tracked"],
            "earned": read.get("trophies_earned"),
            "rates": read["rates"],
        })
    return states


def wls(xs: list[float], ys: list[float], ws: list[float]):
    """Weighted least squares y ~ a + b x. Returns (slope, resid_std) or (None, None)."""
    if len(xs) < 2 or max(xs) - min(xs) < 1e-9:
        return None, None
    sw = sum(ws)
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    if sxx < 1e-12:
        return None, None
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    slope = sxy / sxx
    intercept = my - slope * mx
    dof = max(1, len(xs) - 2)
    var = sum(w * (y - (intercept + slope * x)) ** 2 for x, y, w in zip(xs, ys, ws)) / (sw * dof / len(xs))
    return slope, math.sqrt(max(var, 0.0))


def shrink(slope: float | None, prior: float, n_points: int) -> float:
    if slope is None:
        return prior
    blended = (n_points * slope + SHRINK_K * prior) / (n_points + SHRINK_K)
    return min(SLOPE_MAX, max(SLOPE_MIN, blended))


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    states = refresh_states(case)
    n_states = len(states)
    t0 = states[0]["first_at"]
    target = parse_time(case["target"])
    horizon_days = case["horizon_minutes"] / 1440

    def x_of(t: datetime) -> float:
        return math.log(1 + max(0.0, (t - t0).total_seconds() / 3600))

    xs = [x_of(s["last_at"]) for s in states]
    x_last, x_target = xs[-1], x_of(target)
    dx = max(0.0, x_target - x_last)
    # Recency weights: the newest state counts three times the oldest.
    ws = [1 + 2 * (i / (n_states - 1)) for i in range(n_states)] if n_states > 1 else [1.0]

    last = states[-1]
    players_last = last["players"]
    if not players_last or players_last <= 0:
        raise ValueError(f"{name} has no players tracked in its last read")

    # Earners per trophy per state (rate percent of players tracked), and each state's total.
    earners = []       # earners[s][i], None where the read lacked the trophy
    totals = []        # total earners in state s
    for s in states:
        row = []
        for rate in s["rates"]:
            row.append(None if rate is None else rate * s["players"] / 100)
        earners.append(row)
        totals.append(s["earned"] if s["earned"] else sum(v for v in row if v is not None))
    total_last = totals[-1]

    # Global count growth: players and total earners, each in log space against x. A fitted
    # slope stands as fitted (clamped); the launch priors only fill in when no pair of
    # states identifies one — shrinking a mature fit toward a launch prior would overstate
    # growth and break the count consistency the rescale below relies on.
    slope_p, _ = wls(xs, [math.log(s["players"]) for s in states], ws)
    slope_p = PRIOR_PLAYER_SLOPE if slope_p is None else min(SLOPE_MAX, max(SLOPE_MIN, slope_p))
    slope_tot, _ = wls(xs, [math.log(max(t, 1.0)) for t in totals], ws)
    slope_tot = PRIOR_EARN_SLOPE if slope_tot is None else min(SLOPE_MAX, max(SLOPE_MIN, slope_tot))

    players_pred = players_last * math.exp(min(3.7, slope_p * dx))
    players_pred = max(players_pred, float(players_last))
    total_pred = total_last * math.exp(min(3.9, slope_tot * dx))

    # Per-trophy earners: fit, shrink toward the total slope, extrapolate, floor at the last
    # count (earners do not un-earn), then rescale so the counts sum to the total forecast.
    earn_pred: list[float] = []
    vols: list[float] = []
    anchors: list[float] = []   # each trophy's rate in its last non-null state
    for i in range(len(trophies)):
        pts = [(xs[s], math.log(earners[s][i])) for s in range(n_states)
               if earners[s][i] is not None and earners[s][i] > 0]
        anchor_rate = None
        anchor_earn = None
        for s in range(n_states - 1, -1, -1):
            if states[s]["rates"][i] is not None:
                anchor_rate = states[s]["rates"][i]
                anchor_earn = earners[s][i]
                break
        if anchor_rate is None:            # the trophy never appeared: tier-median prior
            peers = [r for r in last["rates"] if r is not None]
            anchor_rate = sorted(peers)[len(peers) // 2] if peers else 1.0
            anchor_earn = anchor_rate * players_last / 100
        anchors.append(anchor_rate)
        if anchor_earn is None or anchor_earn <= 0:
            anchor_earn = 0.5              # nobody has it yet: half a player of mass
        slope_i, resid = wls([p[0] for p in pts], [p[1] for p in pts],
                             [ws[s] for s in range(n_states)
                              if earners[s][i] is not None and earners[s][i] > 0])
        slope_i = shrink(slope_i, slope_tot, len(pts))
        pred = anchor_earn * math.exp(min(3.9, slope_i * dx))
        earn_pred.append(max(pred, anchor_earn * 0.99))
        vols.append(max(resid, 0.12) if resid is not None else DEFAULT_VOL)
    if total_pred > 0 and sum(earn_pred) > 0:
        factor = min(1.33, max(0.75, total_pred / sum(earn_pred)))
        earn_pred = [e * factor for e in earn_pred]

    # Spread pieces, shared by every trophy: the random walk, refresh uncertainty.
    refresh_sd = 0.45 if n_states == 1 else (0.30 if n_states == 2 else 0.18)
    base_var = (RW_SIGMA * math.sqrt(horizon_days)) ** 2 + refresh_sd ** 2
    # The median may not run away from the last rate without limit.
    logit_cap = 1.2 + 1.1 * math.sqrt(horizon_days)

    rows = []
    for i, trophy in enumerate(trophies):
        rate_last = anchors[i]
        median = min(99.99, max(0.01, 100 * earn_pred[i] / players_pred))
        centre = logit(median)
        if n_states == 1:
            centre -= SINGLE_STATE_DRIFT * math.sqrt(horizon_days)
        lo_cap = logit(rate_last) - logit_cap
        hi_cap = logit(rate_last) + logit_cap
        centre = min(hi_cap, max(lo_cap, centre))
        drift = centre - logit(rate_last)
        sigma = math.sqrt(base_var + (vols[i] * dx) ** 2)
        sigma = min(2.6, max(0.30, sigma))
        sigma_lo = sigma * (1.15 if drift < -0.05 else 0.92 if drift > 0.05 else 1.0)
        sigma_hi = sigma * (0.92 if drift < -0.05 else 1.15 if drift > 0.05 else 1.0)
        qs = []
        for z in Z:
            q = inv_logit(centre + z * (sigma_lo if z < 0 else sigma_hi))
            qs.append(round4(min(100.0, max(0.0, q))))
        for j in range(1, len(qs)):        # rounding must never break the order
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
