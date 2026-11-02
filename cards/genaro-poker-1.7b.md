---
license: apache-2.0
base_model: Qwen/Qwen3-1.7B
base_model_relation: finetune
datasets:
- predbrad/genaro-poker-sft
language:
- en
pipeline_tag: text-generation
library_name: transformers
tags:
- poker
- game-ai
- on-device
- gguf
- llama.cpp
- pokersi
- json
- distillation
---

# Genaro Poker 1.7B

A phone-sized poker player. Genaro is a full fine-tune of
[Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) built to be the opponents in
[**PokerSI**](https://github.com/bradflaugher/PokerSI), an Android poker game where every seat
is a small language model running on the phone with llama.cpp. It is the bigger, sharper
sibling of [Genaro Poker 0.6B](https://huggingface.co/predbrad/Genaro-poker), for phones with 6 GB+ of RAM.

**v3** learns its poker from a **search solver** running on PokerSI's own engine (Monte Carlo
rollouts of every legal move, with Bayesian opponent-range inference), the teacher that beats
PokerSI's House Bot by +256 bb/100. v2 learned from House Bot itself.

Given PokerSI's prompt for one decision (a persona, the rules, what the seat can see, the
legal moves), Genaro answers with one JSON object: a legal move and, on talk turns, a short
line in the character's voice, trained not to give its cards away.

```json
{"action": "call", "amount": 55, "say": "The pot looks like it's brewing a little sunshine."}
{"action": "check", "amount": 0, "say": "Ahoy, these doubloons are worth a fortune!"}
{"action": "raise", "amount": 240, "say": "Get those reps in, I see potential!"}
```

| File | Use |
|---|---|
| `genaro-poker-1.7b-v3-Q4_K_M.gguf` (1.1 GB) | **On-device / llama.cpp.** What PokerSI downloads. Chat template embedded. |
| `model.safetensors` + config/tokenizer (bf16) | transformers, further fine-tuning, other quantizations |
| `genaro-poker-1.7b-Q4_K_M.gguf` | v2, kept so app versions that point at it keep working. Revision `v2` has the v2 weights. |

Smaller, faster sibling for any phone: [predbrad/Genaro-poker](https://huggingface.co/predbrad/Genaro-poker).
Code, solver, data pipeline and evals: [github.com/bradflaugher/genaro-poker](https://github.com/bradflaugher/genaro-poker).
Training data: [predbrad/genaro-poker-sft](https://huggingface.co/datasets/predbrad/genaro-poker-sft).

## Results

600 held-out PokerSI decisions over all six variants (Texas Hold'em, Omaha, Five-Card Draw,
Seven-Card Stud, Short Deck, Crazy Pineapple), each model run through llama.cpp exactly as the
app runs it (same prompt, GBNF grammar and sampling); plus 3,600+ live hands.

<!-- RESULTS -->

- *EV regret*: big blinds a move gives up against the solver's best move, by the solver's own
  EV estimate of every move (the solver's labels themselves score 0.08). *Big mistakes*: moves
  giving up more than 2 bb.
- *Leak*: a line PokerSI's own `HandBlur` would censor. *Card talk*: any card or hand word.
- *Live play*: one seat against three House Bots through PokerSI's engine, paired against
  House Bot playing the same seat on the same deals (which cancels most of the card luck).

All results, including the solver itself: [results](https://github.com/bradflaugher/genaro-poker/blob/main/results/README.md).

## Use

**In PokerSI:** MODELS → Genaro Poker 1.7B → download. That's it.

**llama.cpp:** PokerSI-style messages, with a grammar and thinking off:

```bash
llama-server -m genaro-poker-1.7b-v3-Q4_K_M.gguf --jinja -c 4096
```

The prompt format is PokerSI's (`llm/PokerPrompt.kt`); real examples are in the
[dataset](https://huggingface.co/datasets/predbrad/genaro-poker-sft). Outside that format
Genaro is just a small, poker-obsessed Qwen3; it is not a general chat model.

**transformers:**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("predbrad/Genaro-poker-1.7B")
model = AutoModelForCausalLM.from_pretrained("predbrad/Genaro-poker", dtype="auto")
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
```

## Training

- **Prompts:** self-play hands of PokerSI's own Kotlin engine and prompt builder, compiled into
  a bridge, so every prompt is byte-for-byte what the app sends.
- **Moves:** a decision-time search solver (`bridge/Solver.kt`). For each legal move,
  including several raise sizes and every discard choice, it samples 480 worlds (unseen cards
  reshuffled; each world weighted by how likely the opponents' past actions were with those
  cards), replays the hand, makes the move and plays out with opponent policies drawn from
  House Bot styles plus sticky-caller and sound-regular populations, so it doesn't learn to
  exploit any one of them. It picks the move with the best lower confidence bound on EV
  (noise can't buy a shove), with a bet-size abstraction that rules out giant overbets, and
  lets the persona's style break near-ties. It beats House Bot by +256 ± 49 bb/100.
- **Table talk:** written by Gemma 4 E4B for the chosen move, kept only if it passes the
  grammar's rules, 2-12 words, no card or hand vocabulary, PokerSI's repeat rules and
  PokerSI's `HandBlur`.
- **Recipe:** 35,756 decisions, full fine-tune, bf16, 2 epochs, lr 3e-5 cosine, effective
  batch 64, loss on the reply only. ~4 h on one NVIDIA RTX Spark N1X laptop. See `training_config.yaml`.

## Limits

- Genaro distills the solver but doesn't match it: it gives up about 1.1 bb per decision
  against it; the solver is far ahead of every Genaro in live play.
- The solver best-responds to House-Bot-like play (made robust with other populations); it
  is not a game-theoretic equilibrium solver.
- Trained only for PokerSI's prompt format, in English; lines are family-friendly by
  construction.

## License

Apache 2.0, inherited from Qwen3-1.7B. Training data was generated with PokerSI (MIT) and
Gemma 4 (Apache 2.0).
