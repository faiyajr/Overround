# OVERROUND

**UEFA Champions League prediction and edge engine.** Learned club embeddings trained across Europe's five domestic leagues, feeding a multithreaded C++ Monte Carlo simulation of the 36-team Swiss league phase.

> **Status: pre-alpha, in development.** Nothing below is a measured result yet. Performance and calibration numbers in this README are *targets*, and they will be replaced with verified benchmarks (or revised down) before any of them get claimed anywhere.

Target ship date: **before matchday 1, week of 15 September 2026.** The point is to have predictions on the record ahead of the season, not backtested after it.

---

## The problem

The Champions League is too sparse to model directly. A season is roughly 175 matches, and each club plays only 8 league-phase games. That is nowhere near enough signal to learn team strength from scratch, and it is not obvious how to compare a Bundesliga side to a La Liga side when they barely ever meet.

## The approach

Learn club representations from the dense data, then use the sparse data to align the leagues.

- **Train on ~18,000 matches** across roughly 10 seasons of the Premier League, La Liga, Serie A, Bundesliga and Ligue 1.
- **Use UCL and Europa fixtures as bridges.** Cross-league matches are rare, but they are exactly the observations that calibrate one league's scale against another's.
- **Represent clubs as learned embeddings** rather than a scalar Elo rating. A vector can encode style and matchup structure that a single number cannot, and it projects to 2D for a readable map of European football.

---

## Architecture

Three layers, deliberately separated so each can be benchmarked on its own.

### 1. Model layer (Python / PyTorch)

- Transformer over each club's domestic match sequence, so form is learned rather than hand-coded as rolling averages.
- Learned club embeddings, shared across leagues, projectable to 2D.
- Output is a **full scoreline distribution**, not a W/D/L classification. Every downstream market is derived from that distribution.
- Baselines: Dixon-Coles and Elo, evaluated on the same held-out splits.

### 2. Simulation layer (C++)

- Multithreaded Monte Carlo over the Swiss league phase: 36 clubs, 8 matches each (4 home, 4 away, no repeat opponents), 144 fixtures, goal-difference tiebreakers.
- Finishing position propagates into the seeded bracket. Top 8 go direct to the round of 16, 9th to 24th enter two-legged playoffs, 25th to 36th are eliminated with no Europa parachute. The bracket is fixed from the quarter-finals onward with no re-draw, so complete paths are simulable.
- Also simulates the **constrained draw**: 4 pots of 9 by UEFA coefficient, two opponents drawn per pot, country protection, home/away balance.

### 3. Market layer (v2, in-season)

- Devig bookmaker odds into true implied probabilities.
- Model-vs-market edge detection.
- Fractional Kelly staking.
- Closing line value tracking.
- Markets: qualification, outright winner, 1X2, over/under, BTTS.

### Visuals

36-club embedding map, top-8 / playoff / elimination probability bars, a bracket that fills with simulation mass, and a calibration curve that updates every matchday.

---

## Repository layout

```
engine/         C++ simulation engine
  include/      public headers
  src/
    core/       shared types, fixtures, standings
    draw/       constrained pot draw
    league_phase/  Swiss phase simulation and tiebreakers
    knockout/   seeded bracket and two-legged playoffs
    sampling/   scoreline sampling, lookup tables, RNG
    concurrency/ thread pool, per-thread state
    io/         model artifact loading, result serialization
  bindings/     Python bindings
  bench/        throughput and scaling benchmarks
  tests/        unit tests, draw-constraint property tests

ml/             PyTorch model and evaluation
  overround_ml/
    data/       ingestion, league joins, season splits
    features/   sequence construction, encodings
    models/     embeddings, transformer, scoreline heads
    baselines/  Dixon-Coles, Elo
    train/      training loops, configs, checkpointing
    eval/       RPS, Brier, reliability curves, ablations
    export/     scoreline lookup tables for the engine
  configs/      experiment configs
  notebooks/    exploration only, nothing load-bearing
  tests/

market/         v2: devig, edge, staking, CLV

api/            FastAPI service
  app/
    routers/    endpoints
    services/   simulation orchestration, caching
    schemas/    request/response models
    db/         PostgreSQL access

db/             migrations and seed data
web/            React + D3 front end
data/           raw / interim / processed / external (gitignored)
artifacts/      trained models, lookup tables, sim outputs (gitignored)
scripts/        ingestion, training and deployment entrypoints
docs/           architecture notes and decision records
```

---

## Targets (unverified)

**Simulation throughput**

| Target | Value |
| --- | --- |
| Full-tournament sims | 10,000 in under 2s multithreaded |
| Sustained rate | 5,000+ sims/sec |
| Match samples per tournament sim | ~190 (144 league phase, ~45 knockout) |
| Thread scaling | near-linear to physical core count |

Optimization path if throughput falls short: precompute scoreline distributions, sample from lookup tables, one thread per simulation, then SIMD the sampling loop.

**Predictive calibration**

| Target | Value |
| --- | --- |
| RPS on held-out matches | < 0.21 (sharp closing lines sit around 0.19-0.21) |
| vs Elo baseline | beat on RPS |
| vs Dixon-Coles | match or beat |

**Reported metrics are Brier and RPS, plus sims/sec. Never accuracy.** Accuracy is a vanity metric for anything probability-driven, and it rewards the wrong behaviour on a three-outcome sport with a fat draw rate.

A note on framing: beating a well-calibrated Elo baseline on real football data is genuinely hard. Matching it, staying well calibrated, and being honest about it is still a strong result. This project says *benchmarked against*, not *outperforming*, unless and until the held-out numbers say otherwise.

---

## Scope

**v1 (before matchday 1):** league-phase qualification probabilities only. One model. Deployed.

**After:** knockout bracket simulation, then the market layer during the season.

---

## Key dates (2026/27)

| Event | Date |
| --- | --- |
| Matchday 1 | week of 15 Sep 2026 |
| Matchday 8 | 28 Jan 2027 |
| Knockout playoffs | mid-Feb 2027 |
| Round of 16 | Mar 2027 |
| Final, Allianz Arena | 5 Jun 2027 |

---

## Stack

C++ (simulation engine) · Python · PyTorch · pandas · FastAPI · React · D3.js · PostgreSQL

# Architecture

## Design constraint

The model and the simulator are separated by a **serialized artifact**, not by a function call. The model's job ends when it emits, for every fixture, a probability mass over scorelines. The engine's job starts there and never imports PyTorch.

That boundary buys three things: the C++ side stays a pure sampling problem with no Python in the hot loop, the two halves can be benchmarked independently, and swapping the model (or dropping in Dixon-Coles as a control) means writing a different artifact rather than touching the engine.

## Data flow

```
domestic leagues (~18k matches)  ─┐
UCL / Europa cross-league games  ─┼─> ml/data ─> ml/features ─> ml/train
                                  ┘                                 │
                                                                    v
                                              club embeddings + scoreline model
                                                                    │
                                                        ml/export   v
                                              scoreline lookup tables (artifact)
                                                                    │
                                                                    v
                                        engine: draw ─> league phase ─> knockout
                                                                    │
                                                                    v
                                          aggregated outcome distributions
                                                                    │
                                                      api ─> postgres ─> web
```

## Engine

### core/
Fixture list, club identifiers, standings accumulation, tiebreaker comparator. Standings must be cheap to update in place, so the layout here matters more than it looks: 36 clubs of compact structs, no allocation inside a simulation.

### draw/
The constrained pot draw. 4 pots of 9 by coefficient, two opponents from each pot, country protection, home/away balance. This is a constraint satisfaction problem, not a shuffle, and a naive sequential draw deadlocks on the last few slots. Needs both a rejection-sampling path and a validity check, plus property tests asserting that no generated draw violates a constraint.

### league_phase/
144 fixtures, sampled scorelines, standings, final ordering with goal-difference tiebreakers. Output is a permutation of 36 clubs. This is the only thing v1 needs.

### knockout/
Seeded bracket from the league-phase ordering. Top 8 direct to R16, 9-24 into two-legged playoffs, 25-36 out. Fixed from the quarter-finals, no re-draw, so a full path is deterministic given results.

### sampling/
Scoreline sampling from precomputed distributions, alias-method lookup tables, per-thread RNG. This is the hot loop: ~190 samples per tournament sim, 10,000 sims, so roughly 2M samples per run. If throughput is short, this directory is where the work goes.

### concurrency/
Thread pool, one simulation per thread, per-thread RNG seeded from a single root. No shared mutable state during a simulation; aggregation happens per-thread and merges at the end. That keeps scaling near-linear and keeps results reproducible from a seed.

### io/
Load model artifacts, serialize aggregated results. Kept isolated so the simulation core has no file or format dependencies.

### bindings/
Python bindings so the API and notebooks can drive the engine without a subprocess.

## ML

### data/
Ingestion and normalization across five leagues plus European competition. The hard part is identity: club naming, promotion and relegation, and squad continuity across seasons. Cross-league fixtures need to be tagged explicitly since they carry the calibration signal.

### features/
Per-club match sequences with the leakage rules enforced here, not in the training loop. Splits are chronological. Any random split silently leaks future form into the past and makes every metric meaningless.

### models/
Club embedding table shared across leagues, transformer over match sequences, scoreline head. The head emits a distribution over scorelines, which is what everything downstream consumes.

### baselines/
Dixon-Coles and Elo, run on identical splits with identical evaluation code. These are not decoration. They are the thing the project has to be measured against, and the honest outcome is that they are hard to beat.

### eval/
RPS, Brier, reliability curves, per-league breakdowns, ablations (embeddings vs scalar rating, transformer vs rolling averages, with vs without cross-league bridges). The ablation on the bridges is the one that tests the central claim.

### export/
Convert model outputs into the engine's lookup-table format. Versioned, since a result is only reproducible if the artifact that produced it is identifiable.

## API

FastAPI over PostgreSQL. Endpoints serve stored simulation outputs and model metadata. Simulation runs are triggered as jobs, not per-request, since a full run is seconds of CPU and results are identical for identical inputs. Cache on (model version, fixture state, seed).

## Web

React with D3 for the four visuals: embedding map, qualification bars, bracket mass, calibration curve. The calibration curve updates each matchday and is the honest one, so it stays visible rather than buried.

## Market layer (v2)

Devig, edge detection, fractional Kelly, CLV. Kept as a separate package because it depends on the model but the model must not depend on it. CLV is the metric that matters here, not realized profit over a sample this small.

## Open questions

- Does the transformer beat well-tuned rolling averages once both are properly regularized on 18k matches? Unresolved until the ablation runs.
- How much does the cross-league bridge actually buy, given how few bridge matches exist per season pair?
- Draw simulation cost per tournament sim: if rejection sampling is expensive, does a precomputed pool of valid draws preserve the correct distribution?
- Embedding drift across seasons: fixed per club, or time-varying?
