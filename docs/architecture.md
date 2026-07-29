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
