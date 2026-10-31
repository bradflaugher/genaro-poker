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
---

# Genaro Poker 1.7B

A phone-sized poker player. Genaro is a full fine-tune of
[Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) built to be the opponents in
[**PokerSI**](https://github.com/bradflaugher/PokerSI), an Android poker game where every seat
is a small language model running on the phone with llama.cpp. It is the bigger, sharper
sibling of [Genaro Poker 0.6B](https://huggingface.co/predbrad/Genaro-poker), for phones with 6 GB+ of RAM.

Given PokerSI's prompt for one decision (a persona, the rules, what the seat can see, the
legal moves), Genaro answers with one JSON object: a sound, legal move and, on talk turns, a
short line in the character's voice that never gives its cards away.

```json
{"action": "call", "amount": 55, "say": "The pot looks like it's brewing a little sunshine."}
{"action": "check", "amount": 0, "say": "Ahoy, these doubloons are worth a fortune!"}
{"action": "raise", "amount": 240, "say": "Get those reps in, I see potential!"}
```

| File | Use |
|---|---|
| `genaro-poker-1.7b-Q4_K_M.gguf` (1.1 GB) | **On-device / llama.cpp.** What PokerSI downloads. Chat template embedded. |
| `model.safetensors` + config/tokenizer (bf16) | transformers, further fine-tuning, other quantizations |

Smaller, faster sibling for any phone: [predbrad/Genaro-poker](https://huggingface.co/predbrad/Genaro-poker).
Code, data pipeline and evals: [github.com/bradflaugher/genaro-poker](https://github.com/bradflaugher/genaro-poker).
Training data: [predbrad/genaro-poker-sft](https://huggingface.co/datasets/predbrad/genaro-poker-sft).

## Results

600 held-out PokerSI decisions over all six variants (Texas Hold'em, Omaha, Five-Card Draw,
Seven-Card Stud, Short Deck, Crazy Pineapple). Every model runs through llama.cpp exactly as
the app runs it: same prompt, GBNF grammar and sampling (persona temperature 0.6-0.9,
top-k 40, top-p 0.95, max 96 tokens).

<!-- RESULTS -->

*Agreement*: same move as the majority of 9 House Bot teachers. *Blunder*: folding with
equity ≥ 10 points above the pot odds, or calling/raising with equity ≥ 15 points below
them (bluffs by bluffing personas excepted). *Leak*: a line PokerSI's own `HandBlur` would
censor. *Card talk*: any card, rank or hand word at all. Live play and every other model:
[results](https://github.com/bradflaugher/genaro-poker/blob/main/results/README.md).

## Use

**In PokerSI:** MODELS → Genaro Poker 1.7B → download. That's it.

**llama.cpp:** PokerSI-style messages, with a grammar and thinking off:

```bash
llama-server -m genaro-poker-1.7b-Q4_K_M.gguf --jinja -c 4096
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

- **Prompts:** 33k self-play hands of PokerSI's own Kotlin engine and prompt builder,
  compiled into a bridge, so every prompt is byte-for-byte what the app sends.
- **Moves:** the majority of 9 House Bot teachers (PokerSI's heuristic, each with the
  persona's style and 1,000-sample Monte Carlo equity).
- **Table talk:** written by Gemma 4 E4B for the chosen move, kept only if it passes the
  grammar's rules, 2-12 words, no card or hand vocabulary, PokerSI's repeat rules and
  PokerSI's `HandBlur`. Quiet turns have no line, like the app's quiet-turn grammar.
- **Recipe:** 22,941 conversations, full fine-tune, bf16, 1 epoch, lr 2e-5 cosine, effective
  batch 64, loss on the reply only. ~80 min on one NVIDIA RTX Spark N1X laptop.
  See `training_config.yaml`.

## Limits

- Genaro imitates PokerSI's House Bot. That makes it a reasonable, readable game-night
  opponent, not a strong poker player; in live play it does not beat House Bots.
- Trained only for PokerSI's prompt format, in English.
- Lines are family-friendly by construction; personas come from PokerSI.

## License

Apache 2.0, inherited from Qwen3-1.7B. Training data was generated with PokerSI (MIT) and
Gemma 4 (Apache 2.0).
