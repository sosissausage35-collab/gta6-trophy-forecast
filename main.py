"""GTA VI trophy earn-rate forecaster (make.fun gta6-trophy-rarity-forecast-01).

Reads one case per run (/input/<id>.in, schema makefun/gta6-trophies-case/v1)
and writes /output/<id>.out: five quantiles (5/25/50/75/95%) of each trophy's
earn rate on Exophase at the case's target. Python 3.13 standard library only,
deterministic, no network, no randomness.

Model
-----
An earn rate is a ratio: earners / players_tracked. Exophase refreshes its
counts in batches, so the two move in jumps; the forecaster therefore works
with the two counts separately and divides at the end.

Two regimes, picked from the case's own history:

1. Flat launch night. When every read in the history is identical (same
   players, same rates) and the checkpoint is within a day of release, the
   history carries no trend at all -- it is launch night before Exophase's
   first refresh. The kit's competition.yaml documents that refresh: between
   the 04:00Z and 04:30Z reads on 2026-11-19 the tracked players went
   638 -> 3,868 and the trophies earned 943 -> 4,276, and every rate but
   the one at 100% fell. So a refresh is coming, and its totals are known.
   The new cohort (3,230 players) earned 3,333 trophies; the trophy at
   100% (everyone does it first) accounts for 3,230 of them, leaving 103
   spread over the other 54 trophies. How those 103 spread is the one
   unknown, so the median blends three ways of spreading them (in log
   space, so the blend is a geometric mean):
     - proportional: in proportion to each trophy's old earners;
     - concentrated: all of them on the five highest-rate trophies;
     - early-spread: among the trophies already at 1.10% or more.
   The trophy at 100% stays at 100% in every component. The spread is a
   fixed 0.18 logit units, which covers the disagreement between those
   three spreadings. Validated on simulated refreshes (see README): this
   scores about 0.06-0.13 pinball loss against every plausible refresh,
   against ~1.5 for a persistence forecast.

2. Growth. When the history moves, each trophy's rate is fitted on the
   logit scale against log(1 + hours since release) -- a straight line
   through the last 12 hours of reads, least squares -- and read at the
   target. The logit scale is what keeps a 4.5-day extrapolation from
   running away: a rate that doubles from 1% to 2% moves the same logit
   distance as one that doubles from 33% to 50%, so the fit cannot chase
   a small rate into the sky the way a power law on earners can (that
   version overshot a saturating story trophy to 79% where the truth was
   35%). The logit's move from the last observed rate is capped at 2.0.
   The spread combines the fit's residual scatter with a horizon term
   that grows like the square root of the horizon in days, floored at
   0.25 and capped at 0.80 logit units. A 12-hour window beat 24- and
   36-hour windows on two synthetic final-like histories (a saturating
   one and a still-growing one), and beat both the earners/player power
   law and the persistence baseline on each.

Quantiles are the median shifted by the standard normal's quantile at each
level on the logit scale, rounded to 4 decimals (half up), non-decreasing.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Standard normal quantiles at the levels above (competition.yaml's z_tau).
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# --- Documented launch-night refresh (competition.yaml) ---------------------
# Between the 04:00Z and 04:30Z reads on 2026-11-19, Exophase refreshed the
# game's counts: players tracked 638 -> 3,868, trophies earned 943 -> 4,276.
REFRESH_N0 = 638.0
REFRESH_T0 = 943.0
REFRESH_N1 = 3868.0
REFRESH_T1 = 4276.0
# Blend weights for the three ways of spreading the refresh's extra earners.
BLEND_W = (0.55, 0.25, 0.20)  # proportional, concentrated, early-spread
FLAT_SIGMA = 0.18             # logit spread for a flat-history forecast
FLAT_MAX_AGE_HOURS = 24.0     # flat regime only near release
EARLY_RATE = 1.10             # "early" trophies for the early-spread component

# --- Growth-regime constants ------------------------------------------------
LOGIT_CAP = 2.0        # max logit move from the last observed rate
FIT_WINDOW_HOURS = 12.0  # growth fit uses reads from the last 12 hours
SIGMA_FLOOR = 0.25
SIGMA_CAP = 0.80
SIGMA_HORIZON = 0.20   # per sqrt(day) of horizon, added in quadrature


def clamp_rate(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up (as JavaScript's Math.round does)."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def parse_time(text):
    """Parse an ISO 8601 UTC timestamp, with or without fractional seconds
    (the case's `at`/`target` carry none; history reads carry milliseconds)."""
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_release(when):
    return (when - RELEASE).total_seconds() / 3600.0


def _fit_log_log(xs, ys):
    """Least squares y = a + b*x. Returns (a, b, residual_sd). xs, ys are
    lists of floats of equal length (>= 1)."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0.0 or n == 1:
        return my, 0.0, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    if n <= 2:
        return a, b, 0.0
    resid = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    return a, b, math.sqrt(resid / (n - 2))


def flat_refresh_medians(rates, players, earned):
    """Median forecast for a flat launch-night history.

    rates: last read's rates (percent). players, earned: last read's counts.
    Returns a list of median percents, one per trophy.
    """
    n0 = float(players)
    t0 = float(earned)
    # The documented refresh, scaled to this case's own counts: the ratios
    # are the refresh's signature (players x6.06, trophies x4.53).
    n1 = n0 * (REFRESH_N1 / REFRESH_N0)
    t1 = t0 * (REFRESH_T1 / REFRESH_T0)
    new_players = n1 - n0
    new_earned = t1 - t0
    # Earners per trophy in the last read.
    earners = [r * n0 / 100.0 for r in rates]
    # Trophies at (essentially) 100% stay there: every new player earns them.
    saturated = [r >= 99.99 for r in rates]
    sat_earners = sum(e for e, s in zip(earners, saturated) if s)
    # New earners left for the unsaturated trophies, after the saturated
    # ones take one per new player each.
    n_sat = sum(1 for s in saturated if s)
    remaining = new_earned - new_players * n_sat
    if remaining < 0.0:
        remaining = 0.0
    unsat_earners = t0 - sat_earners

    # Component 1: proportional -- the remaining new earners spread in
    # proportion to each trophy's old earners.
    prop = []
    for r, e, s in zip(rates, earners, saturated):
        if s:
            prop.append(100.0)
        elif unsat_earners > 0:
            prop.append((e + remaining * e / unsat_earners) / n1 * 100.0)
        else:
            prop.append(e / n1 * 100.0)

    # Component 2: concentrated -- all remaining new earners on the five
    # highest-rate unsaturated trophies, the rest pure dilution.
    order = sorted((i for i, s in enumerate(saturated) if not s),
                   key=lambda i: -rates[i])
    top = set(order[:5])
    top_earners = sum(earners[i] for i in top)
    conc = []
    for i, (r, e, s) in enumerate(zip(rates, earners, saturated)):
        if s:
            conc.append(100.0)
        elif i in top and top_earners > 0:
            conc.append((e + remaining * e / top_earners) / n1 * 100.0)
        else:
            conc.append(e / n1 * 100.0)

    # Component 3: early-spread -- the remaining new earners among the
    # trophies already at EARLY_RATE or more, the rest pure dilution.
    early_idx = [i for i, (r, s) in enumerate(zip(rates, saturated))
                 if not s and r >= EARLY_RATE]
    early_earners = sum(earners[i] for i in early_idx)
    early = []
    for i, (r, e, s) in enumerate(zip(rates, earners, saturated)):
        if s:
            early.append(100.0)
        elif i in early_idx and early_earners > 0:
            early.append((e + remaining * e / early_earners) / n1 * 100.0)
        else:
            early.append(e / n1 * 100.0)

    # Geometric blend of the three components, floored so a trophy nobody
    # has earned yet still gets a small positive median.
    medians = []
    for p, c, e in zip(prop, conc, early):
        parts = [max(0.02, p), max(0.02, c), max(0.02, e)]
        lg = sum(w * math.log(v) for w, v in zip(BLEND_W, parts))
        medians.append(min(99.99, math.exp(lg)))
    return medians


def growth_medians_and_sigmas(case):
    """Median + sigma per trophy from the history's growth.

    For each trophy, fits logit(rate) as a straight line in log(1 + hours
    since release) over the last FIT_WINDOW_HOURS of reads (all reads if
    fewer than three fall in the window) and reads the line at the target.
    The logit scale is what keeps a 4.5-day extrapolation honest: it
    compresses growth, so a rate cannot run away the way a power law on
    earners can. The logit's move from the last observed rate is capped.
    """
    trophies = case["trophies"]
    history = case["history"]
    target = parse_time(case["target"])
    t_target = hours_since_release(target)
    horizon_days = case["horizon_minutes"] / 1440.0
    x_target = math.log1p(max(0.0, t_target))

    t_last = hours_since_release(parse_time(history[-1]["at"]))
    medians, sigmas = [], []
    for i in range(len(trophies)):
        last_rate = None
        for read in reversed(history):
            if i < len(read["rates"]) and read["rates"][i] is not None:
                last_rate = read["rates"][i]
                break
        if last_rate is None:
            medians.append(0.5)
            sigmas.append(SIGMA_CAP)
            continue
        xs, ys = [], []
        for read in history:
            rates = read["rates"]
            if i >= len(rates) or rates[i] is None:
                continue
            t = hours_since_release(parse_time(read["at"]))
            if t_last - t <= FIT_WINDOW_HOURS:
                xs.append(math.log1p(max(0.0, t)))
                ys.append(logit(rates[i]))
        if len(xs) < 3:
            xs, ys = [], []
            for read in history:
                rates = read["rates"]
                if i >= len(rates) or rates[i] is None:
                    continue
                t = hours_since_release(parse_time(read["at"]))
                xs.append(math.log1p(max(0.0, t)))
                ys.append(logit(rates[i]))
        if not xs:
            medians.append(last_rate)
            sigmas.append(SIGMA_CAP)
            continue
        a, b, resid = _fit_log_log(xs, ys)
        lg = a + b * x_target
        lg_last = logit(last_rate)
        if lg > lg_last + LOGIT_CAP:
            lg = lg_last + LOGIT_CAP
        elif lg < lg_last - LOGIT_CAP:
            lg = lg_last - LOGIT_CAP
        median = inv_logit(lg)
        sigma = math.sqrt(resid * resid +
                          (SIGMA_HORIZON * math.sqrt(horizon_days)) ** 2)
        sigma = min(SIGMA_CAP, max(SIGMA_FLOOR, sigma))
        medians.append(median)
        sigmas.append(sigma)
    return medians, sigmas


def history_is_flat(case):
    """True when every read in the history is identical (same players and
    same rates for every trophy)."""
    history = case["history"]
    if not history:
        return False
    first = history[0]
    for read in history[1:]:
        if read.get("players_tracked") != first.get("players_tracked"):
            return False
        if read.get("trophies_earned") != first.get("trophies_earned"):
            return False
        if read.get("rates") != first.get("rates"):
            return False
    return True


def forecast_case(case):
    trophies = case["trophies"]
    history = case["history"]
    if not history:
        # No data at all: a flat, wide forecast rather than a crash.
        return {"trophies": [{"id": t["id"], "q": [0.1, 0.3, 0.5, 0.9, 1.6]}
                             for t in trophies]}
    last = history[-1]
    rates = last["rates"]
    checkpoint = parse_time(case["at"])
    age_hours = hours_since_release(checkpoint)

    if history_is_flat(case) and age_hours <= FLAT_MAX_AGE_HOURS:
        medians = flat_refresh_medians(rates, last["players_tracked"],
                                       last["trophies_earned"])
        sigmas = []
        for r in rates:
            # A trophy nobody has earned yet is far more uncertain.
            sigmas.append(0.80 if (r or 0.0) < 0.05 else FLAT_SIGMA)
    else:
        medians, sigmas = growth_medians_and_sigmas(case)

    rows = []
    for trophy, median, sigma in zip(trophies, medians, sigmas):
        lg = logit(median)
        qs = [round4(inv_logit(lg + z * sigma)) for z in Z]
        # Rounding must not break the ordering.
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        qs = [min(100.0, max(0.0, q)) for q in qs]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        out = OUTPUT_DIR / (path.stem + ".out")
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n",
                       encoding="utf-8")


if __name__ == "__main__":
    main()
