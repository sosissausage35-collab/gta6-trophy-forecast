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
100%, and the remaining new earners are split three ways and blended
geometrically (weights 0.55 / 0.25 / 0.20):

- *proportional* — new earners in proportion to each trophy's old earners;
- *concentrated* — the extra earners go to the five highest-rate trophies;
- *early-spread* — they go to trophies already at ≥1.10% (story progress
  and the fishing challenge), the ones new players actually reach first.

The blend's median is floored at 0.02%. Spread is 0.18 logit units (0.80
for trophies below 0.05%).

Why the blend: the refresh arithmetic is exact — 3,230 new players, 3,333
new earners, and the 100% trophy accounts for 3,230 of them, leaving only
103 extra earners over the other 54 trophies. A proportional split fits
those totals with a power law (new-cohort earn probability ∝ rate^1.6–1.9),
but the split across trophies is genuinely uncertain, so the blend hedges
it instead of pretending to know it.

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

| simulated truth | loss | typical miss | 90% coverage |
|---|---|---|---|
| proportional | 0.070 | 0.025 pp | 0.89 |
| power-law γ=1.6 | 0.057 | 0.018 pp | 0.98 |
| concentrated top-5 | 0.130 | 0.057 pp | 0.89 |
| early-spread | 0.106 | 0.037 pp | 0.82 |
| pure dilution | 0.130 | 0.044 pp | 0.89 |

Average over the four plausible truths: **0.091**, worst **0.130**. For
reference, the dev leaderboard at the time of writing has rank 1 at 0.081,
rank 10 at 0.122, rank 20 at 0.131, rank 50 at 0.154 — so this sits around
the top 10–20 on the plausible truths, top ~1–3 if the refresh is close to
proportional.

Synthetic final-like cases (2-day history with batch refreshes, 4.5-day
horizon, simulated truth): **loss 0.117** (saturating truth) and **0.094**
(still-growing truth), versus 0.251 / 0.304 for the persistence baseline.

## Weak spots, honestly

- The real practice answers are unknown; every practice number above is
  against a simulated refresh, not the graders' truth.
- The final horizon (4.5 days) is 72× the practice horizon (90 min). The
  growth regime is validated only on synthetic histories I constructed
  myself, not on real data.
- If the refresh's extra earners concentrate very differently from the
  three-way blend (e.g. all on one mid-tier trophy), the practice loss
  could be ~0.13 instead of ~0.06.
- The spread on the synthetic finals covers 100% of the truth — the
  synthetic truths are smooth, so that is not a reliable calibration; the
  chosen σ (~0.45 for the final) is an honest guess at 4.5-day uncertainty,
  not a measured one.

---

<sub>Entered on [make.fun](https://make.fun/bounty/gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
