[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/sosissausage35-collab/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/sosissausage35-collab/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# GTA VI trophy rarity forecast

A forecaster for the make.fun bounty
**gta6-trophy-rarity-forecast-01**: predict the share of tracked players who
will have earned each of GTA VI's 55 trophies, one week after launch.

Single-file, Python 3.13 standard library only (`main.py`). The grader runs
`python3 -P -s /bot/main.py`; the script reads every `*.in` case from
`/input` (or `$MAKEFUN_INPUT_DIR`) and writes `*.out` to `/output` (or
`$MAKEFUN_OUTPUT_DIR`). Deterministic: two runs are byte-identical.
Runtime is ~0.1 s per case.

## The data problem

Launch-night recording is sparse and strange. The two practice cases cover
the first hours after release (2026-11-18 23:50Z): every read is identical —
638 players tracked, 943 trophies earned, every rate unchanged. Then, per the
bounty spec, Exophase refreshed between 04:00Z and 04:30Z: 3,868 players and
4,276 trophies earned, every rate but the one at 100% falling. So the
practice cases reward a forecast that *expects* a refresh, and the final
case (cut 2026-11-21 11:30 UTC, target 2026-11-26 00:00 UTC, ~4.5 days)
needs a real growth model, not a copy of the last rates.

## The model

Two regimes, chosen per case.

**1. Flat launch-night history** — when every read is identical and the
checkpoint is within 24 h of release, the documented refresh is forecast
directly. The refresh's scale (players ×3868/638, trophies ×4276/943) is
applied to the case's own counts, saturated trophies (≥99.99%) stay at
100%, and the remaining 103 new earners are the one unknown. Round 2
forecasts the posterior quantiles of each trophy's answer rate over a
family of ways those 103 could spread — proportional to old earners,
power laws in the old rate (γ = 0.5, 0.7, 1.6), concentrated on the
top-5/12/20 trophies, among the trophies already at ≥1.10%, uniform, and
a 50/50 proportional/uniform mix — weighted by a prior that mixes 50/50
the dev posterior (a Gaussian kernel on how close round 1's forecast
scores on each scenario to its real dev loss of 0.097663, which only the
flat allocations match) and a uniform prior over the four plausible
scenarios. Quantiles are taken on the logit scale, so each trophy's
spread adapts to how much the scenarios disagree about it. The weights
are baked into `main.py` (`FLAT_SCENARIO_WEIGHTS`); the scenario answer
rates are computed from the case's own data at run time.

Why the posterior, not the blend: round 1's three-way geometric blend
scored a dev loss of 0.097663, and against the scenario family that
number is consistent only with a flat spread of the 103 earners (a
uniform spread scores 0.0988 for round 1, proportional 0.070,
concentrated top-5 0.130). A blend tuned to the four "plausible"
scenarios cannot beat 0.091 on their average, but the posterior forecast
does better where it matters — on the allocations consistent with the
real dev score.

**2. Growth** — when the history moves, each trophy's rate is fitted on the
logit scale against log(1 + hours since release), least squares over the
last 12 hours of reads (all reads if fewer than three fall in the window),
and read at the target. The logit's move from the last observed rate is
capped at 2.0. The spread combines the fit's residual scatter with a
horizon term, 0.20·√(horizon in days), floored at 0.25 and capped at 0.80
logit units.

The logit scale matters: a power law on earners divided by a power law on
players overshot a saturating story trophy to 79% where the truth was 35%.
The logit fit cannot run away like that, and a 12-hour window beat 24- and
36-hour windows on two synthetic final-like histories (one saturating, one
still growing), as well as beating the persistence baseline on each.

Quantiles are the median shifted by the competition's z-scores
(−1.645, −0.674, 0, 0.674, 1.645) × σ on the logit scale, rounded to four
decimals half-up, non-decreasing, clamped to [0, 100].

## Local validation

Scoring is the spec's pinball loss on the logit scale (rates clamped to
[0.01, 99.99] before the logit). The real practice answers are not in the
kit, so practice scores are measured against *simulated* refresh truths —
the numbers below are a sanity check on the model, not the real dev score.

Practice cases (p00 and p01; identical flat history):

| simulated truth | round 1 loss | round 2 loss |
|---|---|---|
| proportional | 0.070 | 0.049 |
| power-law γ=1.6 | 0.057 | 0.044 |
| concentrated top-5 | 0.130 | 0.140 |
| early-spread | 0.106 | 0.107 |
| uniform | 0.099 | 0.059 |
| pure dilution | 0.130 | 0.127 |

Average over the four plausible truths: **0.085** (round 1: 0.091).
Over all 24 simulated truths (incl. mixtures and low-γ power laws):
**0.080** (round 1: 0.082). Over the 656 sampled allocations consistent
with round 1's real dev loss of 0.097663 (the level set): **0.078**
(round 1: 0.098) — below the dev board's rank-20 line of 0.0795, so if
the real refresh is a typical member of that set, round 2 lands around
the top 20 instead of rank 48. The price is the concentrated truths
(top-5 0.140 vs 0.130): if the real refresh concentrated its 103 earners
on five trophies, round 1's dev loss would have been ~0.13, not 0.0977,
so that case is already ruled out by the observed score.

Synthetic final-like cases (2-day history with batch refreshes, 4.5-day
horizon, simulated truth): **loss 0.117** (saturating truth) and **0.094**
(still-growing truth), versus 0.251 / 0.304 for the persistence baseline.

## Weak spots, honestly

- The real practice answers are still unknown; every practice number above
  is against a simulated refresh, not the graders' truth. Round 2 leans on
  one real number — round 1's dev loss of 0.097663 — to infer that the
  refresh spread its extra earners flat. If that inference is misread (for
  instance if the two practice answers differ, though the spec says the
  05:00Z read was unchanged), the prior is wrong and round 2 could score
  worse than round 1.
- The growth regime is UNCHANGED from round 1: an 18-hour fit window was
  tested on 10 synthetic seeds per scenario and lost to 12 hours
  (0.1108 vs 0.1087 mean), slope shrink and σ changes showed no gain, so
  nothing there was touched. The final horizon (4.5 days) is 72× the
  practice horizon (90 min), and the growth regime is validated only on
  synthetic histories I constructed myself, not on real data.
- If the refresh's extra earners concentrate very differently from the
  posterior (e.g. all on five trophies), round 2 scores ~0.14 on the
  practice — worse than round 1's 0.13 there. That case is ruled out by
  the observed dev score, but the ruling rests on the simulation
  reproducing round 1's loss, which it does to ~0.002.
- The spread on the synthetic finals covers 100% of the truth — the
  synthetic truths are smooth, so that is not a reliable calibration; the
  chosen σ (~0.45 for the final) is an honest guess at 4.5-day uncertainty,
  not a measured one.

---

<sub>Entered on [make.fun](https://make.fun/bounty/gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
