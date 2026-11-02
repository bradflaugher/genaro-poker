# genaro-poker

**Genaro** is a phone-sized LLM that plays poker. It is fine-tuned for one job: being the
opponents in [PokerSI](https://github.com/bradflaugher/PokerSI), the retro Android poker
game where every seat is a small language model running on the phone with llama.cpp.

| | Genaro Poker 0.6B | Genaro Poker 1.7B |
|---|---|---|
| Download | [predbrad/Genaro-poker](https://huggingface.co/predbrad/Genaro-poker) | [predbrad/Genaro-poker-1.7B](https://huggingface.co/predbrad/Genaro-poker-1.7B) |
| GGUF (Q4_K_M) | 397 MB | 1.1 GB |
| For | any phone that runs PokerSI (the app's default) | 6 GB+ phones |
| Base | [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) (Apache 2.0) | [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) (Apache 2.0) |

Training data: [predbrad/genaro-poker-sft](https://huggingface.co/datasets/predbrad/genaro-poker-sft).
Current version: **v3**, distilled from a search solver that runs on PokerSI's own engine.

## Why fine-tune for one app

PokerSI asks a tiny model to do a lot in 96 tokens: read the table, pick a legal move inside a
GBNF grammar, and say one short, in-character line **without giving its cards away**. Stock
small models get the JSON right (the grammar sees to that) but play loosely and talk badly:

- **They leak.** Stock Qwen3 names its cards or hand in 10-18% of its lines ("Let's take the
  Aces and raise!"). PokerSI's hand blur censors them, but a censor bar every few turns isn't fun.
- **They ramble and parrot.** Half of Qwen3 1.7B's lines run past the 12 words the prompt asks
  for; Qwen3 0.6B repeats itself in 24% of lines.
- **They play loose**, often just following the prompt's one-word advice.

Genaro is trained on PokerSI's **exact** prompts and grammar: solver-quality moves, a line in
the persona's voice, nothing about its cards.

## Results

600 held-out decisions across all six variants, each model run through llama.cpp exactly as
PokerSI runs it (the app's prompt, grammar and sampling), plus thousands of live hands.
Everything, including the solver itself and every experiment: [`results/README.md`](results/README.md).

| model | GGUF | EV regret (bb/decision) | big mistakes /100 | leaks | card talk | repeats | >12 words | live bb/100 (same 1,200 deals) | live vs House Bot, paired (3,600 hands) |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3 1.7B (base) | 1282 MB | 1.64 | 18.2 | 17.9% | 49.7% | 8.2% | 48.3% | — | — |
| LFM2.5 1.2B (PokerSI's previous default) | 731 MB | 1.31 | 16.0 | 2.6% | 29.8% | 8.5% | 9.4% | -80 ± 80 | — |
| Genaro Poker 1.7B v2 (House Bot labels) | 1107 MB | 1.29 | 17.2 | 0.3% | 0.9% | 0.6% | 0.0% | -45 ± 64 | — |
| **Genaro Poker 0.6B v3** | 397 MB | 1.24 | 13.2 | 0.3% | 0.6% | 0.0% | 0.0% | +61 ± 59 | +44 ± 46 |
| **Genaro Poker 1.7B v3** | 1107 MB | 1.11 | 10.8 | 0.3% | 1.4% | 0.3% | 0.0% | +95 ± 62 | +30 ± 46 |

- **EV regret**: big blinds a move gives up against the solver's best move, by the solver's
  own EV estimate of every move. The solver's labels themselves score 0.08. **Big mistakes**:
  moves giving up more than 2 bb.
- **Leak**: a line PokerSI's own `HandBlur` would censor. **Card talk**: any card or hand word.
- **Live play**: one seat against three House Bots (PokerSI's built-in heuristic) through
  PokerSI's engine, 100 bb deep, all six variants. *Paired* is the difference from House Bot
  playing the same seat on the same deals, which cancels most of the card luck.

**The short version.** v3 (solver labels) plays better than v2 (House Bot labels) and
LFM2.5, the previous default: about 140 bb/100 better live on the same deals, a quarter fewer
big mistakes, and the cleanest table talk. Against House Bot head to head it is ahead but
within the noise (+30 to +44 ± 46). The solver it learned from is far stronger
(+256 ± 49), so there's room left.

## How it's built

```
PokerSI (Kotlin) ──bridge──▶ self-play: every decision's exact prompt + grammar
                                 │
            search solver ───────┤ the move: Monte Carlo rollouts on PokerSI's engine
  Gemma 4 E4B + PokerSI HandBlur ┤ one clean table-talk line per talk turn
                                 ▼
          35.8k decisions ──▶ full fine-tune of Qwen3 ──▶ GGUF Q4_K_M ──▶ PokerSI
```

1. **No reimplementation.** `bridge/Bridge.kt` is compiled *together with PokerSI's own*
   engine, prompt, grammar, persona, table-talk and hand-blur code (`bridge/build.sh`). Every
   training prompt is byte-for-byte what the app sends, and evaluation uses the app's own
   reply parser and leak detector.
2. **The solver** (`bridge/Solver.kt`). A true equilibrium (CFR) solver covers heads-up
   Hold'em; PokerSI has six variants and up to six players. So Genaro's teacher is
   decision-time search, the approach multiway poker bots use:
   - **Rollouts:** for each legal move (several raise sizes, every discard choice), sample 480
     worlds. Each world keeps every card the player can see where it is in the deck and
     reshuffles the rest, replays the hand so far (checked: replays reproduce the real hand
     exactly), makes the move and plays it out. All moves see the same worlds (common random
     numbers).
   - **Range inference:** each world is weighted by how likely the opponents' recorded actions
     were with the cards it gave them, so a raiser is credited with a strong hand.
   - **Robust opponents:** rollouts play opponents from a population (their own House Bot
     style, sticky callers, sound regulars), so it can't win by exploiting House Bot's
     over-folding.
   - **Bet-size abstraction:** all-ins only when they aren't giant overbets or the stack is
     short. Without it the solver found silly 50×-pot shoves; with it, it won more.
   - **Lower confidence bound:** a move must beat checking or folding by more than its
     standard error, so noise can't buy an aggressive play. The persona's style
     (aggression, tightness, bluffing) breaks near-ties, so characters keep their flavour.

   It costs about 70 ms per decision and beats House Bot by **+256 ± 49 bb/100** (paired,
   3,600 hands), ahead in every variant.
3. **Table talk** comes from Gemma 4 E4B, given the full prompt and the chosen move. Each line
   must pass, in order: the grammar's character rules, 2-12 words, no card or hand vocabulary
   at all, PokerSI's `TableTalk` repeat rules, and PokerSI's own `HandBlur`. When the solver
   changed a talk turn's move, the line is rewritten for the new move.
4. **Training:** full fine-tune in bf16, loss on the reply only, thinking disabled in the chat
   template (as PokerSI renders it). The 0.6B repeats decisions by how much EV is at stake
   (cost-sensitive SFT).
5. **Export:** `convert_hf_to_gguf.py` + `llama-quantize` to Q4_K_M with Genaro metadata and
   the chat template embedded (PokerSI reads the template from the GGUF).
6. **Evaluation:** offline EV regret against the solver, then live hands through PokerSI's
   engine against House Bots, paired on identical deals.

## What worked and what didn't

| change | effect |
|---|---|
| One House Bot sample → majority of 9 (v1 → v2) | +100 bb/100 live (−184 → −84) |
| House Bot labels → solver labels (v2 → v3) | ~+140 bb/100 live, big mistakes 18.5 → 13.8 /100 |
| Second epoch on 1.7B | 68% → 80% teacher agreement (v2) |
| Cost-sensitive weighting (0.6B) | +28 ± 25 bb/100 live over plain v3 (7,200 paired hands): likely, not significant. Shipped |
| DAgger: the solver labels 4,500 states the 0.6B itself reached | 0 ± 35 live; no measurable gain |
| Deciding at low temperature instead of the persona's | no gain (regret 1.18 → 1.16): sampling isn't the bottleneck |

## Reproduce

Needs: an NVIDIA GPU (24 GB is plenty for 0.6B), a PokerSI checkout, and llama.cpp in
`~/Work/llama.cpp` (`build/` CPU; `build-cuda/` optional, for `GENARO_GPU=1` fast evals).

```bash
make setup            # uv env (torch, transformers, TRL) + JDK 21 / Kotlin via mise
make data             # compile the bridge against PokerSI, self-play 33k hands
make teacher          # table talk from Gemma 4 E4B (~1.5 h)
make solver-play      # check the solver beats House Bot before trusting it as a teacher
make solve            # solver labels for 36k decisions (~1 h)
make relabel-talk     # rewrite lines whose move the solver changed (~1 h)
make sft-data train   # build the set, full fine-tune (SIZE=0.6b or 1.7b)
make gguf eval play report
```

Every step is a small module in `genaro/`; `configs/` holds the training settings.

## Limits

- **Genaro doesn't match its teacher.** It gives up about 1.1-1.2 bb per decision against the
  solver; the solver beats House Bot by +256, Genaro by a noisy +30-44.
- **The solver best-responds** to House-Bot-like play (made robust with other populations);
  it is not a game-theoretic equilibrium.
- **Short contexts only.** Trained on PokerSI prompts (≤ 900 tokens); it is not a chat model.
- **English only**, family-friendly lines, inherited from the prompts it learned from.

## Versions

| | labels | Hugging Face |
|---|---|---|
| v3 | search solver | `main`; GGUF `genaro-poker-<size>-v3-Q4_K_M.gguf` |
| v2 | majority of 9 House Bots | revision `v2`; GGUF `genaro-poker-<size>-Q4_K_M.gguf` (kept for installed app versions) |

## License

Code: MIT. Model weights: Apache 2.0 (from Qwen3). Training data: Apache 2.0 (generated
with PokerSI, MIT, and Gemma 4, Apache 2.0).
