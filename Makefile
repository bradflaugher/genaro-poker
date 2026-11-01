# Genaro: a phone-sized poker model for PokerSI. Full pipeline, top to bottom.
# uv run --no-sync: a wheel-tag quirk in nvidia-cusparselt makes uv reinstall it on every
# sync on aarch64. Sync once with `make setup`, then skip it.
PY := uv run --no-sync python
POKERSI ?= $(HOME)/Work/PokerSI
SIZE ?= 0.6b

.PHONY: setup bridge data votes teacher sft-data train gguf eval play report dataset guard

setup:                # Python env (torch cu134, transformers, TRL) + JDK/Kotlin for the bridge
	uv sync
	mise install

bridge:               # compile the bridge against PokerSI's own engine + prompt code
	POKERSI=$(POKERSI) mise exec -- bridge/build.sh

data: bridge          # self-play every variant; one record per decision (~15 min)
	mkdir -p data
	mise exec -- java -jar bridge/bridge.jar dump --hands 30000 --seed 1 --out data/train-decisions.jsonl
	mise exec -- java -jar bridge/bridge.jar dump --hands 3000 --seed 2 --out data/test-decisions.jsonl

teacher:              # table talk from Gemma 4 E4B, filtered by PokerSI's HandBlur (~1.5 h)
	$(PY) -m genaro.teacher --split train --n 12000 --batch 128 --k 2 --teacher google/gemma-4-E4B-it

votes:                # 9-teacher majority labels for the training + eval rows (same records, ~20 min)
	$(PY) -m genaro.ids
	mise exec -- java -jar bridge/bridge.jar dump --hands 30000 --seed 1 --out data/train-decisions-voted.jsonl --vote-ids data/vote-ids.txt
	mise exec -- java -jar bridge/bridge.jar dump --hands 3000 --seed 2 --out data/test-decisions-voted.jsonl --vote-ids data/test-vote-ids.txt

sft-data:
	$(PY) -m genaro.build --model Qwen/Qwen3-0.6B --extra-quiet 12000 --out sft-qwen3-maj

train:                # full fine-tune (~70 min for 0.6B, ~2 h for 1.7B on the N1X)
	$(PY) -m genaro.sft configs/genaro-$(SIZE).yaml

gguf:                 # Q4_K_M with Genaro metadata (needs llama.cpp built in ~/Work/llama.cpp)
	$(PY) -m genaro.export runs/genaro-$(SIZE)/final out/genaro-poker-$(SIZE)-Q4_K_M.gguf --size $(shell echo $(SIZE) | tr a-z A-Z)

eval:                 # 600 held-out decisions through llama.cpp with the app's grammar + sampling
	$(PY) -m genaro.evaluate out/genaro-poker-$(SIZE)-Q4_K_M.gguf --name genaro-poker-$(SIZE)

play:                 # 1,200 live hands vs House Bots through PokerSI's engine
	$(PY) -m genaro.play out/genaro-poker-$(SIZE)-Q4_K_M.gguf --name genaro-poker-$(SIZE) --hands 1200 --workers 12

report:
	$(PY) -m genaro.report

dataset:              # Parquet export for the Hugging Face dataset repo
	$(PY) -m genaro.dataset

guard:                # unified-memory watchdog for long jobs on the N1X
	setsid nohup scripts/memguard.sh > memguard.log 2>&1 < /dev/null &
