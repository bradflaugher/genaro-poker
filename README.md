# genaro-poker

**Genaro** is a phone-sized LLM that plays poker. It is fine-tuned for one job: being the
opponents in [PokerSI](https://github.com/bradflaugher/PokerSI), the retro Android poker
game where every seat is a small language model running on the phone with llama.cpp.

| | Genaro 0.6B | Genaro 1.7B |
|---|---|---|
| Download | [predbrad/Genaro-poker](https://huggingface.co/predbrad/Genaro-poker) | [predbrad/Genaro-poker-1.7B](https://huggingface.co/predbrad/Genaro-poker-1.7B) |
| GGUF (Q4_K_M) | 397 MB | 1.1 GB |
| For | any phone that runs PokerSI (the app's default) | 6 GB+ phones |
| Base | [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) (Apache 2.0) | [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) (Apache 2.0) |

Training data: [predbrad/genaro-poker-sft](https://huggingface.co/datasets/predbrad/genaro-poker-sft).

## Why fine-tune for one app

PokerSI asks a tiny model to do a lot in 96 tokens: read the table, pick a legal move inside a
GBNF grammar, and say one short, in-character line **without giving its cards away**. Stock
small models get the JSON right (the grammar sees to that) but play loosely and talk badly:

- **They leak.** Stock Qwen3 names its cards or hand in 10-18% of its lines ("Let's take the
  Aces and raise!"). PokerSI's hand blur censors them, but a censor bar every few turns isn't fun.
- **They ramble and parrot.** Half of Qwen3 1.7B's lines run past the 12 words the prompt asks
  for; Qwen3 0.6B repeats itself in 24% of lines.
- **They follow the prompt's one-word advice** even when the pot odds say otherwise.

Genaro is trained on PokerSI's **exact** prompts and grammar, so it learns the house style:
sound decisions, a line in the persona's voice, nothing about its cards.

## Results

600 held-out decisions across all six variants, each model run through llama.cpp exactly as
PokerSI runs it (the app's prompt, grammar and sampling). Full tables, including live play:
[`results/README.md`](results/README.md).

| model | GGUF | agreement | blunders /100 | leaks | card talk | repeats | >12 words | live play bb/100 |
|---|---|---|---|---|---|---|---|---|
| Qwen3 0.6B (base) | 484 MB | 46.0% | 10.3 | 10.5% | 39.8% | 23.9% | 22.7% | — |
| Qwen3 1.7B (base) | 1282 MB | 39.7% | 8.2 | 17.9% | 49.7% | 8.2% | 48.3% | — |
| LFM2.5 1.2B (PokerSI's previous default) | 731 MB | 55.8% | 12.6 | 2.6% | 29.8% | 8.5% | 9.4% | −80 ± 80 |
| **Genaro Poker 0.6B** | **397 MB** | **78.5%** | 9.1 | **0.0%** | **0.3%** | **0.0%** | **0.0%** | −84 ± 52 |
| **Genaro Poker 1.7B** | 1107 MB | **80.2%** | **7.5** | 0.3% | 0.9% | 0.6% | **0.0%** | **−45 ± 64** |

*Agreement*: same move as the majority of 9 House Bot teachers. *Blunder*: folding with equity
≥ 10 points above the pot odds, or calling/raising with equity ≥ 15 points below them. *Leak*:
a line PokerSI's own `HandBlur` would censor; *card talk*: any card or hand word at all.
*Live play*: 1,200 hands in one seat against three House Bots (which win +80 ± 66 in the same
seat), identical deals for every model.

The short version: Genaro 0.6B plays about as well as the previous default in live hands
(well inside the error bars) at half the download, and its table talk is clean: no leaks, no
rambling, half the repeats. Genaro 1.7B is the strongest model on every decision metric and
the best LLM in live play, for phones that can afford 1.1 GB. The biggest single win was the
labels: training on one noisy teacher sample (v1) lost 184 bb/100 live; the 9-teacher
majority cut that to 84. For 1.7B, a second epoch mattered (68% → 80% agreement).

## How it's built

```
PokerSI (Kotlin)  ──bridge──▶  self-play: every decision's exact prompt + grammar + equity
                                   │
             9 House Bot teachers ─┤  majority move (1,000-sample Monte Carlo equity each)
   Gemma 4 E4B + PokerSI HandBlur ─┤  one clean table-talk line per talk turn
                                   ▼
            22.9k conversations ──▶ full fine-tune of Qwen3 ──▶ GGUF Q4_K_M ──▶ PokerSI
```

1. **No reimplementation.** `bridge/Bridge.kt` is compiled *together with PokerSI's own*
   engine, prompt, grammar, persona, table-talk and hand-blur code (`bridge/build.sh`). Every
   training prompt is byte-for-byte what the app sends, and evaluation uses the app's own
   reply parser and leak detector.
2. **Decisions** come from PokerSI's House Bot heuristic with each persona's style and a
   sharper equity estimate. One sample is noisy (it agrees with a 9-teacher majority only 87%
   of the time), so labels are the **majority of 9 independent teachers**.
3. **Table talk** comes from Gemma 4 E4B, given the full prompt and the chosen move. Each line
   must pass, in order: the grammar's character rules, 2-12 words, no card or hand vocabulary
   at all, PokerSI's `TableTalk` repeat rules, and PokerSI's own `HandBlur`. Quiet turns carry
   no line, exactly like the app's quiet-turn grammar.
4. **Training:** full fine-tune in bf16, loss on the reply only, thinking disabled in the chat
   template (as PokerSI renders it). About 70 minutes for 0.6B and 2 hours for 1.7B on one
   [RTX Spark N1X laptop](https://github.com/bradflaugher/train-me).
5. **Export:** `convert_hf_to_gguf.py` + `llama-quantize` to Q4_K_M with Genaro metadata and
   the chat template embedded (PokerSI reads the template from the GGUF).
6. **Evaluation:** offline decisions vs. equity and the teacher majority; then live hands
   through PokerSI's engine against House Bots, with identical deals for every model.

## Reproduce

Needs: an NVIDIA GPU (24 GB is plenty for 0.6B), a PokerSI checkout, and llama.cpp built in
`~/Work/llama.cpp` (`llama-server`, `llama-quantize`).

```bash
make setup            # uv env (torch, transformers, TRL) + JDK 21 / Kotlin via mise
make data             # compile the bridge against PokerSI, self-play 33k hands
make teacher          # table talk from Gemma 4 E4B (~1.5 h)
make votes            # 9-teacher majority labels (~20 min)
make sft-data train   # build the set, full fine-tune (SIZE=0.6b or 1.7b)
make gguf eval play report
```

Every step is a small module in `genaro/`; `configs/` holds the training settings.

## Limits

- **Genaro imitates a heuristic.** Its ceiling is PokerSI's House Bot, not good poker. In live
  play it doesn't beat House Bots (see the results). For a game-night opponent that is fine;
  a stronger teacher (a real solver) is the obvious next step.
- **Short contexts only.** Trained on PokerSI prompts (≤ 900 tokens); it is not a chat model.
- **English only**, family-friendly lines, inherited from the prompts it learned from.

## License

Code: MIT. Model weights: Apache 2.0 (from Qwen3). Training data: Apache 2.0 (generated
with PokerSI, MIT, and Gemma 4, Apache 2.0).
