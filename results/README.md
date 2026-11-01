# Results

## Decisions and table talk (600 held-out PokerSI decisions, all six variants)

Every model runs through llama.cpp exactly as PokerSI runs it: the app's prompt, its GBNF grammar and its sampling (persona temperature 0.6-0.9, top-k 40, top-p 0.95). *Agreement* is with the majority of 9 House Bot teachers using 1,000-sample equity. A *blunder* is folding with equity at least 10 points above the pot odds, or calling/raising with equity at least 15 points below them (bluff raises by bluffing personas excepted). *Leaks* are lines PokerSI's own HandBlur would censor.

| model | GGUF | agreement | blunders /100 | bad folds /100 | blunder cost (bb/100 decisions) | leaks | card talk | repeats | >12 words | tokens/reply |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3 0.6B (base) | 484 MB | 46.0% | 10.3 | 5.9 | 14.2 | 10.5% | 39.8% | 23.9% | 22.7% | 24.9 |
| Qwen3 1.7B (base) | 1282 MB | 39.7% | 8.2 | 3.6 | 7.9 | 17.9% | 49.7% | 8.2% | 48.3% | 28.4 |
| LFM2.5 1.2B (PokerSI's previous default) | 731 MB | 55.8% | 12.6 | 9.9 | 10.9 | 2.6% | 29.8% | 8.5% | 9.4% | 23.0 |
| Genaro 0.6B v1 (single-sample labels) | 397 MB | 75.0% | 7.6 | 5.3 | 18.7 | 0.0% | 0.3% | 0.0% | 0.0% | 21.7 |
| Genaro 1.7B v1 (single-sample labels) | 1107 MB | 68.2% | 9.4 | 8.0 | 12.1 | 0.0% | 1.1% | 0.3% | 0.0% | 21.9 |
| Genaro 1.7B v2 (majority labels, 1 epoch) | 1107 MB | 68.3% | 9.4 | 8.0 | 12.7 | 0.0% | 1.7% | 0.3% | 0.0% | 21.8 |
| **Genaro 0.6B** (majority labels, 2 epochs) | 397 MB | 78.5% | 9.1 | 6.7 | 13.5 | 0.0% | 0.3% | 0.0% | 0.0% | 21.7 |
| **Genaro 1.7B** (majority labels, 2 epochs) | 1107 MB | 80.2% | 7.5 | 5.7 | 9.2 | 0.3% | 0.9% | 0.6% | 0.0% | 21.6 |

## Live play (1,200 hands, identical deals for every row)

The model plays one seat of a 4-handed table against three House Bots through PokerSI's real engine (`bridge/Bridge.kt play`), all six variants in rotation, 100 big blinds deep. Win rate has a large standard error at this sample size; the operational numbers are the robust ones.

| seat 0 played by | bb/100 (± s.e.) | fallback to House Bot | leaks | repeats | words/line | tokens/reply |
|---|---|---|---|---|---|---|
| House Bot (the app's heuristic) | +80 ± 66 | — | — | — | — | — |
| Always follow the prompt's advice | -50 ± 79 | — | — | — | — | — |
| LFM2.5 1.2B | -80 ± 80 | 0.00% | 2.1% | 17.4% | 8.0 | 23.7 |
| Genaro 0.6B v1 | -184 ± 53 | 0.07% | 0.0% | 11.4% | 6.8 | 22.0 |
| **Genaro 0.6B** | -84 ± 52 | 0.18% | 0.0% | 8.7% | 6.9 | 22.0 |
| **Genaro 1.7B** | -45 ± 64 | 0.00% | 0.0% | 6.8% | 6.8 | 22.1 |
