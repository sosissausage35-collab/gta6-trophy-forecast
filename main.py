"""GTA VI trophy rarity forecast, round 4: a hedge (make.fun
gta6-trophy-rarity-forecast-01).

One run reads every ``*.in`` case from /input (or $MAKEFUN_INPUT_DIR)
and writes ``*.out`` to /output (or $MAKEFUN_OUTPUT_DIR): for each of
the game's 55 trophies, five quantiles (5/25/50/75/95%) of the share
of tracked players who will have earned it at the case target.
Python 3.13 standard library only, deterministic, no network, no
randomness; the grader runs ``python3 -P -s /bot/main.py``.

The practice cases are flat launch-night histories (every read the
same: 638 players, 943 trophies earned), and the bounty spec says
Exophase refreshed between the 04:00Z and 04:30Z reads on 2026-11-19
-- players 638 -> 3,868, trophies earned 943 -> 4,276. The refresh's
totals are known; the only unknown is how the 103 extra earners
(3,333 new earners minus 3,230 that the one trophy already at 100%
absorbs) spread over the other 54 trophies. So a flat history is
forecast by computing the refresh directly, under ten different
spreads of those 103 earners, and the final case (cut 2026-11-21
11:30 UTC, target 2026-11-26 00:00 UTC) needs the growth regime
below instead.

The hedge
---------
Two earlier rounds disagreed about that spread. Round 1 blended
three spreads geometrically -- proportional to old earners (0.55),
concentrated on the five highest-rate trophies (0.25), among the
trophies already at 1.10% or more (0.20) -- and scored a real dev
loss of 0.097663. Round 2 took posterior quantiles over the same
ten spreads, weighted by how well round 1 scores on each of them
plus a uniform prior, and scored worse: 0.101441. The truth sits
between the two views -- the blend is too flat, the posterior too
concentrated -- so this round averages the two forecasts in logit
space, one quantile at a time:

    q = sigmoid((logit(q_round1) + logit(q_round2)) / 2)

A logit-space midpoint of two quantile forecasts is itself a proper
quantile forecast, and it can never repeat either parent's failure
mode: near either parent it scores near the better one, between
them it scores better than both.

How the code is organised
--------------------------
``refresh_context`` scales the documented refresh to the case's own
counts and works out what is left to spread. ``refresh_table`` builds
one answer-rate row per spread from a small spec list -- proportional,
three power laws in the old rate, concentrated on the top 5/12/20,
early, uniform, and a half-proportional half-uniform mix -- so the
two parent views are two readings of one table, not two programs.
Round 1's view is the geometric blend of three rows; round 2's is
the weight-weighted quantile of the rows on the logit scale; the
hedge is their midpoint. The growth regime fits logit(rate) as a
line in log(1 + hours since release) over the last 12 hours of
reads, caps the move from the last observed rate at 2.0 logit
units, and spreads by the fit's residual scatter plus a horizon
term, floored at 0.25 and capped at 0.80.

Local validation (trophy-r2 harness, simulated refresh truths):
the hedge averages 0.0869 over the four plausible truths (round 1:
0.0912, round 2: 0.0850) and 0.1483 worst; on the 656 sampled
allocations consistent with round 1's real dev loss it averages
0.0848 (round 1: 0.0984, round 2: 0.0777), beats round 1 on 93.8%
of them, and its worst is 0.1120 against round 2's 0.1249. The
growth regime is shared code and scores identically to both
parents on synthetic final-like cases (0.1027-0.1124). Local
numbers are optimistic -- round 2's local 0.0777 became a real
0.1014 -- and the hedge's value is the midpoint property above,
not the local mean.
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

# The documented launch-night refresh: between the 04:00Z and 04:30Z
# reads on 2026-11-19, Exophase's counts went players 638 -> 3,868
# and trophies earned 943 -> 4,276.
REFRESH_N0 = 638.0
REFRESH_T0 = 943.0
REFRESH_N1 = 3868.0
REFRESH_T1 = 4276.0

# Flat regime: round 1's geometric blend (proportional / concentrated
# top-5 / early-spread), its logit spread, and how long after release
# a flat history is still treated as the refresh.
BLEND_W = (0.55, 0.25, 0.20)
FLAT_SIGMA = 0.18
FLAT_WIDE_SIGMA = 0.80  # for a trophy nobody has earned yet
FLAT_MAX_AGE_HOURS = 24.0
BLEND_FLOOR = 0.02  # floor under a blend component, in percent
EARLY_RATE = 1.10  # "early" trophies sit at this rate or above

# Round 2's posterior weights over the refresh spreads, computed
# offline from round 1's dev evidence: a 50/50 mix of a Gaussian
# kernel (width 0.008) on |round-1 loss on the spread - 0.097663|
# and a uniform prior over the four plausible spreads (prop,
# gamma1.6, conc5, early).
FLAT_SCENARIO_WEIGHTS = {
    "uniform": 0.252378,
    "early": 0.223083,
    "prop": 0.125001,
    "gamma1.6": 0.125,
    "conc5": 0.125,
    "conc20": 0.108996,
    "gamma0.5": 0.020241,
    "mix_prop+uniform": 0.014617,
    "conc12": 0.004539,
    "gamma0.7": 0.001105,
}

# Growth regime: the most the logit may move from the last observed
# rate, the window the fit looks back over, and the spread's floor,
# cap, and per-sqrt(day) horizon term.
LOGIT_CAP = 2.0
FIT_WINDOW_HOURS = 12.0
SIGMA_FLOOR = 0.25
SIGMA_CAP = 0.80
SIGMA_HORIZON = 0.20

# The ten spreads of the refresh's extra earners, as (name, kind,
# parameter) specs. kinds: "earners" weights by old earners, "power"
# by rate**g, "top" concentrates on the k highest-rate unsaturated
# trophies, "early" keeps only trophies already at EARLY_RATE, "flat"
# weights every unsaturated trophy equally.
SCENARIO_SPECS = [
    ("prop", "earners", None),
    ("gamma0.5", "power", 0.5),
    ("gamma0.7", "power", 0.7),
    ("gamma1.6", "power", 1.6),
    ("conc5", "top", 5),
    ("conc12", "top", 12),
    ("conc20", "top", 20),
    ("early", "early", None),
    ("uniform", "flat", None),
]


def logit(pct):
    """Logit of a percent, clamped to [0.01, 99.99] first."""
    p = min(99.99, max(0.01, pct)) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    """Percent whose logit is lg, computed without overflow either way."""
    if lg < 0:
        e = math.exp(lg)
        return 100.0 * e / (1.0 + e)
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    """Round to 4 decimals, half up (matching JavaScript's Math.round)."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def read_stamp(text):
    """Parse an ISO 8601 UTC timestamp into an aware datetime.

    The case's ``at``/``target`` carry no fractional seconds; history
    reads carry milliseconds. ``fromisoformat`` handles both once the
    trailing Z is swapped for +00:00; the strptime formats stay as a
    fallback for anything it rejects.
    """
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        pass
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"unparsable timestamp: {text!r}")


def hours_after_release(when):
    return (when - RELEASE).total_seconds() / 3600.0


def refresh_context(rates, players, earned):
    """Scale the documented refresh to one case's own last read.

    Returns the refreshed player count, each trophy's old earners,
    which trophies are saturated (at 99.99% or more), the extra
    earners still to spread, and the unsaturated trophy indices.
    The trophy at 100% takes one new earner per new player, so the
    remainder is what the spreads argue over.
    """
    n0 = float(players)
    t0 = float(earned)
    n1 = n0 * (REFRESH_N1 / REFRESH_N0)
    t1 = t0 * (REFRESH_T1 / REFRESH_T0)
    added_players = n1 - n0
    added_earned = t1 - t0
    earners = [r * n0 / 100.0 for r in rates]
    saturated = [r >= 99.99 for r in rates]
    n_sat = sum(1 for s in saturated if s)
    remaining = max(0.0, added_earned - added_players * n_sat)
    return {
        "n1": n1,
        "earners": earners,
        "saturated": saturated,
        "remaining": remaining,
        "unsat": [i for i, s in enumerate(saturated) if not s],
    }


def apply_spread(ctx, weights):
    """Answer rates (%) if the refresh's leftover earners are spread
    over the unsaturated trophies with the given per-trophy weights.

    Saturated trophies stay at 100%. An empty weight set spreads
    nothing (pure dilution); a zero total weight falls back to an
    even split over the weighted trophies.
    """
    total = sum(weights.values())
    n1 = ctx["n1"]
    remaining = ctx["remaining"]
    out = []
    for i, (e, s) in enumerate(zip(ctx["earners"], ctx["saturated"])):
        if s:
            out.append(100.0)
        elif not weights:
            out.append(e / n1 * 100.0)
        elif total <= 0.0:
            out.append((e + remaining / len(weights)) / n1 * 100.0)
        else:
            out.append((e + remaining * weights.get(i, 0.0) / total)
                       / n1 * 100.0)
    return out


def scenario_weights(spec, ctx, rates):
    """Per-trophy weights for one spread spec, over unsaturated
    trophies only (a dict index -> weight)."""
    _name, kind, param = spec
    unsat = ctx["unsat"]
    earners = ctx["earners"]
    if kind == "earners":
        return {i: earners[i] for i in unsat}
    if kind == "power":
        return {i: (rates[i] / 100.0) ** param for i in unsat}
    if kind == "top":
        best = sorted(unsat, key=lambda i: -rates[i])[:param]
        return {i: earners[i] for i in best}
    if kind == "early":
        return {i: earners[i] for i in unsat if rates[i] >= EARLY_RATE}
    return {i: 1.0 for i in unsat}


def refresh_table(rates, players, earned):
    """One answer-rate row per spread, from a single refresh context.

    The mix row is built from the prop and uniform rows after the
    loop: half proportional, half uniform.
    """
    ctx = refresh_context(rates, players, earned)
    table = {}
    for spec in SCENARIO_SPECS:
        table[spec[0]] = apply_spread(
            ctx, scenario_weights(spec, ctx, rates))
    prop = table["prop"]
    unif = table["uniform"]
    table["mix_prop+uniform"] = [0.5 * p + 0.5 * u
                                 for p, u in zip(prop, unif)]
    return table


def round1_centres(table):
    """Round 1's view: the geometric blend of the prop / conc5 /
    early rows of the scenario table, floored so a trophy nobody has
    earned yet still gets a small positive centre."""
    rows = (table["prop"], table["conc5"], table["early"])
    centres = []
    for vals in zip(*rows):
        floored = [v if v > BLEND_FLOOR else BLEND_FLOOR for v in vals]
        lg = 0.0
        for w, v in zip(BLEND_W, floored):
            lg += w * math.log(v)
        centres.append(min(99.99, math.exp(lg)))
    return centres


def weighted_tau(pairs, tau):
    """The tau-quantile of (value, weight) pairs, sorted by value:
    the first value whose cumulative weight reaches tau of the
    total; the largest value if it never does."""
    ordered = sorted(pairs)
    total = 0.0
    for _, w in ordered:
        total += w
    acc = 0.0
    for value, w in ordered:
        acc += w
        if acc >= tau * total:
            return value
    return ordered[-1][0]


def round2_quantiles(table):
    """Round 2's view: for each trophy, the FLAT_SCENARIO_WEIGHTS-
    weighted quantiles of its scenario answer rates on the logit
    scale, so the spread adapts to how much the spreads disagree
    about that trophy."""
    # Only spreads actually in the table take part; their weights are
    # renormalised, so a table missing a row still gives a proper
    # weighted quantile instead of a KeyError.
    present = [name for name in FLAT_SCENARIO_WEIGHTS if name in table]
    total = sum(FLAT_SCENARIO_WEIGHTS[name] for name in present)
    n = len(next(iter(table.values())))
    out = []
    for i in range(n):
        pairs = []
        for name in present:
            pairs.append((logit(table[name][i]),
                          FLAT_SCENARIO_WEIGHTS[name] / total))
        out.append([inv_logit(weighted_tau(pairs, tau))
                    for tau in LEVELS])
    return out


def finish(qs):
    """Round to 4 decimals, keep non-decreasing, clamp to [0, 100]."""
    rounded = [round4(q) for q in qs]
    out = []
    prev = 0.0
    for q in rounded:
        if q < prev:
            q = prev
        out.append(q)
        prev = q
    return [min(100.0, max(0.0, q)) for q in out]


def logit_midpoint(a_qs, b_qs):
    """Logit-space midpoint of two quantile lists, level by level:
    sigmoid((logit(a) + logit(b)) / 2). The hedge itself, factored
    out so the averaging step reads as one operation."""
    return [inv_logit((logit(a) + logit(b)) / 2.0)
            for a, b in zip(a_qs, b_qs)]


def flat_forecast(trophies, rates, players, earned):
    """Five hedged quantiles per trophy for a flat launch-night
    history.

    Both parents come from the same scenario table: round 1's
    geometric blend (with its 0.18 logit spread, 0.80 for a trophy
    nobody has earned yet) and round 2's posterior quantiles. Each
    level's hedge is the logit-space midpoint of the two parents'
    quantiles at that level.
    """
    table = refresh_table(rates, players, earned)
    centres = round1_centres(table)
    post = round2_quantiles(table)
    rows = []
    for trophy, centre, post_qs, last in zip(
            trophies, centres, post, rates):
        sigma = FLAT_WIDE_SIGMA if (last or 0.0) < 0.05 else FLAT_SIGMA
        centre_lg = logit(centre)
        r1 = [inv_logit(centre_lg + z * sigma) for z in Z]
        hedge = logit_midpoint(r1, post_qs)
        rows.append({"id": trophy["id"], "q": finish(hedge)})
    return rows


def _fit_log_log(xs, ys):
    """Least squares y = a + b*x over the given points.

    Returns (intercept, slope, residual_sd): the residual scatter is
    0 when the fit is exact or there are too few points to judge.
    """
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = 0.0
    sxy = 0.0
    for x, y in zip(xs, ys):
        dx = x - mx
        sxx += dx * dx
        sxy += dx * (y - my)
    if sxx <= 0.0 or n == 1:
        return my, 0.0, 0.0
    b = sxy / sxx
    a = my - b * mx
    if n <= 2:
        return a, b, 0.0
    sse = 0.0
    for x, y in zip(xs, ys):
        r = y - (a + b * x)
        sse += r * r
    return a, b, math.sqrt(sse / (n - 2))


def growth_forecast(case):
    """Five quantiles per trophy from a history that actually moves.

    For each trophy, logit(rate) is fitted as a straight line in
    log(1 + hours since release) over the last FIT_WINDOW_HOURS of
    reads (all reads if fewer than three fall in the window) and read
    at the target. The logit scale is what keeps a 4.5-day
    extrapolation honest: it compresses growth, so a rate cannot run
    away the way a power law on earners can. The logit's move from
    the last observed rate is capped. The spread combines the fit's
    residual scatter with a horizon term that grows like the square
    root of the horizon in days.
    """
    trophies = case["trophies"]
    history = case["history"]
    target = read_stamp(case["target"])
    x_target = math.log1p(max(0.0, hours_after_release(target)))
    horizon_days = case["horizon_minutes"] / 1440.0
    # Parse each read's timestamp once: the per-trophy loops below
    # used to re-parse the same stamps 55 times over.
    stamped = [(hours_after_release(read_stamp(read["at"])), read["rates"])
               for read in history]
    t_last = stamped[-1][0]

    rows = []
    for i, trophy in enumerate(trophies):
        # The trophy's last observed rate, scanning backwards.
        last_rate = None
        for _t, rates in reversed(stamped):
            if i < len(rates) and rates[i] is not None:
                last_rate = rates[i]
                break
        if last_rate is None:
            rows.append({"id": trophy["id"],
                         "q": finish([inv_logit(z * SIGMA_CAP)
                                      for z in Z])})
            continue

        # (x, y) points: log-hours against logit(rate), recent window.
        pts = []
        for t, rates in stamped:
            if i >= len(rates) or rates[i] is None:
                continue
            if t_last - t <= FIT_WINDOW_HOURS:
                pts.append((math.log1p(max(0.0, t)), logit(rates[i])))
        if len(pts) < 3:
            pts = [(math.log1p(max(0.0, t)), logit(rates[i]))
                   for t, rates in stamped
                   if i < len(rates) and rates[i] is not None]
        if not pts:
            rows.append({"id": trophy["id"],
                         "q": finish([inv_logit(logit(last_rate)
                                                + z * SIGMA_CAP)
                                      for z in Z])})
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        a, b, resid = _fit_log_log(xs, ys)
        lg = a + b * x_target
        lg_last = logit(last_rate)
        if lg > lg_last + LOGIT_CAP:
            lg = lg_last + LOGIT_CAP
        elif lg < lg_last - LOGIT_CAP:
            lg = lg_last - LOGIT_CAP
        sigma = math.sqrt(resid * resid +
                          (SIGMA_HORIZON * math.sqrt(horizon_days)) ** 2)
        if sigma < SIGMA_FLOOR:
            sigma = SIGMA_FLOOR
        elif sigma > SIGMA_CAP:
            sigma = SIGMA_CAP
        rows.append({"id": trophy["id"],
                     "q": finish([inv_logit(lg + z * sigma) for z in Z])})
    return rows


def reads_identical(case):
    """True when every read in the history is identical -- same
    players tracked, same trophies earned, same rates for every
    trophy. An empty history is not identical; a single read is."""
    history = case["history"]
    if len(history) < 2:
        return bool(history)

    def key(read):
        return (read.get("players_tracked"),
                read.get("trophies_earned"),
                tuple(read.get("rates") or []))

    first = key(history[0])
    return all(key(read) == first for read in history[1:])


def forecast_case(case):
    """Route one case to the flat hedge or the growth forecast."""
    trophies = case["trophies"]
    history = case["history"]
    if not history:
        # No data at all: a flat, wide forecast rather than a crash.
        return {"trophies": [{"id": t["id"], "q": [0.1, 0.3, 0.5, 0.9, 1.6]}
                             for t in trophies]}
    last = history[-1]
    checkpoint = read_stamp(case["at"])
    age = hours_after_release(checkpoint)
    if age <= FLAT_MAX_AGE_HOURS and reads_identical(case):
        # Flat launch night: the hedge of the two flat views.
        return {"trophies": flat_forecast(
            trophies, last["rates"], last["players_tracked"],
            last["trophies_earned"])}
    return {"trophies": growth_forecast(case)}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        with path.open(encoding="utf-8") as fh:
            case = json.load(fh)
        forecast = forecast_case(case)
        out_path = OUTPUT_DIR / (path.stem + ".out")
        out_path.write_text(json.dumps(forecast, separators=(",", ":")) + "\n",
                            encoding="utf-8")


if __name__ == "__main__":
    main()
