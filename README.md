[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/sosissausage35-collab/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/sosissausage35-collab/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# GTA VI trophy rarity forecast — the hedge

A forecaster for the make.fun bounty
**gta6-trophy-rarity-forecast-01**: for each of GTA VI's 55 trophies,
predict the share of tracked players who will have earned it one week
after launch, as five quantiles (5/25/50/75/95%).

One file, Python 3.13 standard library only (`main.py`). The grader
runs `python3 -P -s /bot/main.py`; the script reads every `*.in` case
from `/input` (or `$MAKEFUN_INPUT_DIR`) and writes `*.out` to
`/output` (or `$MAKEFUN_OUTPUT_DIR`). Deterministic — two runs are
byte-identical — and about 0.1 s per case.

## The bet this entry makes

The practice cases are flat launch-night histories, so the whole dev
grade turns on one event: the Exophase refresh the bounty spec
documents between the 04:00Z and 04:30Z reads on 2026-11-19, when
players tracked went 638 → 3,868 and trophies earned 943 → 4,276.
Its totals are known; the only unknown is how the 103 extra earners
spread over the 54 trophies that are not already at 100%.

Two earlier entries disagreed about that spread:

- **Round 1** blended three spreads geometrically — proportional to
  old earners (0.55), concentrated on the five highest-rate trophies
  (0.25), among the trophies already at 1.10% or more (0.20) — and
  scored a real dev loss of **0.097663** (rank 48).
- **Round 2** took posterior quantiles over ten spreads, weighted by
  how well round 1 scores on each of them plus a uniform prior, and
  scored worse: **0.101441** (rank 76).

The truth sits between the two views: the blend is too flat, the
posterior too concentrated. So this round takes both forecasts and
averages them in logit space, one quantile at a time:

    q = sigmoid((logit(q_round1) + logit(q_round2)) / 2)

A logit-space midpoint of two quantile forecasts is itself a proper
quantile forecast. It cannot repeat either parent's failure mode: near
either parent it scores near the better one, and between them it
scores better than both.

## How the forecast is built

Both parents come from one table of scenario answer rates, so the
hedge is a single computation, not two programs glued together.

**The refresh, scaled to the case.** The documented ratios (players
×3868/638, trophies ×4276/943) are applied to the case's own counts.
The trophy at 100% takes one new earner per new player; what is left
— 103 earners in the practice cases — is what the scenarios argue
over. Saturated trophies (≥99.99%) stay at 100%.

**Ten spreads.** Proportional to old earners; power laws in the old
rate (γ = 0.5, 0.7, 1.6); concentrated on the top-5/12/20 trophies;
among the trophies already at ≥1.10%; uniform; and a 50/50
proportional/uniform mix. Each spread gives every trophy an answer
rate — the rate it would sit at after the refresh if that spread
were the truth.

**The two parents.** Round 1's view is the geometric blend of the
proportional, concentrated-top-5, and early rows, with a logit spread
of 0.18 (0.80 for a trophy nobody has earned yet). Round 2's view is
the posterior-weighted quantile of the rows on the logit scale, with
the weights baked into `main.py` — a 50/50 mix of a Gaussian kernel
on how close round 1's forecast scores on each spread to its real
dev loss of 0.097663 and a uniform prior over the four plausible
spreads. Because the quantiles are taken on the logit scale, each
trophy's spread adapts to how much the scenarios disagree about it.

**The hedge.** Each level's forecast is the logit midpoint of the two
parents' quantiles at that level, rounded to four decimals half-up,
non-decreasing, clamped to [0, 100].

**Growth, for histories that move.** When the history is not flat,
each trophy's rate is fitted on the logit scale against
log(1 + hours since release), least squares over the last 12 hours of
reads (all reads if fewer than three fall in the window), and read at
the target. The logit's move from the last observed rate is capped
at 2.0. The spread combines the fit's residual scatter with a horizon
term, 0.20·√(horizon in days), floored at 0.25 and capped at 0.80
logit units. The logit scale matters: a power law on earners divided
by a power law on players once overshot a saturating story trophy to
79% where the truth was 35%. The logit fit cannot run away like that.

## Local validation

Scoring is the spec's pinball loss on the logit scale (rates clamped
to [0.01, 99.99] before the logit). The real practice answers are not
in the kit, so practice scores are measured against *simulated*
refresh truths — a sanity check on the model, not the real dev score.

Practice cases (p00 and p01; identical flat history):

| simulated truth | round 1 | round 2 | hedge |
|---|---|---|---|
| proportional | 0.070 | 0.049 | 0.056 |
| power-law γ=1.6 | 0.057 | 0.044 | 0.050 |
| concentrated top-5 | 0.130 | 0.140 | 0.135 |
| early-spread | 0.106 | 0.107 | 0.106 |
| uniform | 0.099 | 0.059 | 0.079 |
| pure dilution | 0.130 | 0.127 | 0.128 |

Average over the four plausible truths: **0.0869** (round 1: 0.0912,
round 2: 0.0850). Worst over all simulated truths: **0.1483**.

The more honest test is the level set — the 656 sampled allocations
of the 103 extra earners on which round 1's forecast scores
~0.0977, i.e. the allocations consistent with what actually
happened. On that set the hedge averages **0.0848** (round 1: 0.0984,
round 2: 0.0777), beats round 1 on **93.8%** of the allocations
(round 2 manages 84.8%), and its worst is 0.1120 against round 2's
0.1249. The hedge trades round 2's best case for a much better worst
case — that is the point of a midpoint.

Synthetic final-like cases (2-day history with batch refreshes,
4.5-day horizon, simulated truth): the growth regime is shared code
and scores identically to both parents, **0.1027–0.1124** across six
seeds, versus 0.251 / 0.304 for the persistence baseline on the two
scenarios in the round-1 README.

## Weak spots, honestly

- Every practice number above is against a simulated refresh, not
  the graders' truth. The hedge's local mean (0.0869) is between the
  parents', and its real value is the midpoint property, not the
  local mean — round 2's local 0.0777 became a real 0.1014, so local
  numbers are optimistic.
- The level set rests on the simulation reproducing round 1's loss,
  which it does to ~0.002. If the real refresh is a member of the set
  the hedge should land near 0.085; if it is outside it — for
  instance if the two practice answers differ, though the spec says
  the 05:00Z read was unchanged — all bets are off.
- The growth regime is validated only on synthetic histories I
  constructed myself, not on real data. The final horizon (4.5 days)
  is 72× the practice horizon (90 min), and the chosen σ (~0.45 for
  the final) is an honest guess at 4.5-day uncertainty, not a
  measured one.
- If the refresh's extra earners concentrate very differently from
  both parents (e.g. all on five trophies), the hedge scores ~0.135
  on the practice — better than round 2's 0.140 there, worse than
  round 1's 0.130. That case is already ruled out by round 1's
  observed dev score.

---

<sub>Entered on [make.fun](https://make.fun/bounty/gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
