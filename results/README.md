# Results

## Decisions and table talk (600 held-out PokerSI decisions, all six variants)

Every model runs through llama.cpp exactly as PokerSI runs it: the app's prompt, its GBNF grammar and its sampling (persona temperature 0.6-0.9, top-k 40, top-p 0.95).

- *EV regret*: big blinds the move gives up against the solver's best move, from the solver's own EV estimate of every move (480 rollouts each, `bridge/Solver.kt`). *Big mistakes*: moves that give up more than 2 bb. Lower is better.
- *Solver / House Bot agreement*: same move word as the solver, or as the majority of 9 House Bots.
- *Leaks*: lines PokerSI's own HandBlur would censor. *Card talk*: any card or hand word at all.

| model | GGUF | EV regret (bb/decision) | big mistakes /100 | solver agreement | House Bot agreement | leaks | card talk | repeats | >12 words |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3 0.6B (base) | 484 MB | 1.75 | 19.5 | 48.0% | 46.0% | 10.5% | 39.8% | 23.9% | 22.7% |
| Qwen3 1.7B (base) | 1282 MB | 1.64 | 18.2 | 43.2% | 39.7% | 17.9% | 49.7% | 8.2% | 48.3% |
| LFM2.5 1.2B (PokerSI's previous default) | 731 MB | 1.31 | 16.0 | 54.3% | 55.8% | 2.6% | 29.8% | 8.5% | 9.4% |
| Genaro 0.6B v1 (one House Bot sample) | 397 MB | 1.49 | 19.3 | 50.0% | 75.0% | 0.0% | 0.3% | 0.0% | 0.0% |
| Genaro 0.6B v2 (House Bot majority) | 397 MB | 1.41 | 18.5 | 51.8% | 78.5% | 0.0% | 0.3% | 0.0% | 0.0% |
| Genaro 1.7B v2 (House Bot majority) | 1107 MB | 1.29 | 17.2 | 51.2% | 80.2% | 0.3% | 0.9% | 0.6% | 0.0% |
| Genaro 0.6B v3, plain SFT (solver labels) | 397 MB | 1.18 | 13.8 | 64.7% | 56.7% | 0.0% | 0.6% | 0.0% | 0.0% |
| Genaro 0.6B v3 + DAgger (solver labels) | 397 MB | 1.16 | 13.0 | 65.7% | 54.3% | 0.3% | 0.3% | 0.3% | 0.0% |
| **Genaro 0.6B v3**, EV-weighted (solver labels, shipped) | 397 MB | 1.24 | 13.2 | 67.3% | 52.8% | 0.3% | 0.6% | 0.0% | 0.0% |
| **Genaro 1.7B v3** (solver labels, shipped) | 1107 MB | 1.11 | 10.8 | 67.0% | 53.0% | 0.3% | 1.4% | 0.3% | 0.0% |

## Live play

One seat of a 4-handed table against three House Bots through PokerSI's real engine (`bridge/Bridge.kt play`), all six variants in rotation, 100 big blinds deep, identical deals for every row. *vs House Bot* is the paired difference against House Bot playing the same seat on the same deals, which cancels most of the card luck; raw win rates are much noisier.

| seat 0 played by | hands | bb/100 (± s.e.) | vs House Bot, paired (bb/100) | leaks | repeats | words/line |
|---|---|---|---|---|---|---|
| House Bot (the app's heuristic) | 3,600 | +19 ± 38 | — | — | — | — |
| The solver itself (the teacher) | 3,600 | +275 ± 42 | +256 ± 49 | — | — | — |
| Always follow the prompt's advice | 1,200 | -50 ± 79 | — | — | — | — |
| LFM2.5 1.2B | 1,200 | -80 ± 80 | — | 2.1% | 17.4% | 8.0 |
| Genaro 0.6B v1 | 1,200 | -184 ± 53 | — | 0.0% | 11.4% | 6.8 |
| Genaro 0.6B v2 | 1,200 | -84 ± 52 | — | 0.0% | 8.7% | 6.9 |
| Genaro 1.7B v2 | 1,200 | -45 ± 64 | — | 0.0% | 6.8% | 6.8 |
| Genaro 0.6B v3, plain SFT | 3,600 | +21 ± 30 | +1 ± 44 | 0.2% | 6.9% | 6.8 |
| Genaro 0.6B v3 + DAgger | 3,600 | +21 ± 35 | +1 ± 46 | 0.1% | 6.7% | 6.6 |
| **Genaro 0.6B v3**, EV-weighted (shipped) | 3,600 | +63 ± 36 | +44 ± 46 | 0.1% | 5.8% | 6.7 |
| **Genaro 1.7B v3** (shipped) | 3,600 | +50 ± 36 | +30 ± 46 | 0.1% | 5.7% | 6.9 |
