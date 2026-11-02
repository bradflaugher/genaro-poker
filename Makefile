# Genaro: a phone-sized poker model for PokerSI. Full pipeline, top to bottom.
# uv run --no-sync: a wheel-tag quirk in nvidia-cusparselt makes uv reinstall it on every
# sync on aarch64. Sync once with `make setup`, then skip it.
PY := uv run --no-sync python
POKERSI ?= $(HOME)/Work/PokerSI
SIZE ?= 0.6b

.PHONY: setup bridge data votes solve relabel-talk sft-data train gguf eval play solver-play dagger report dataset guard

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

solve:                # solver labels (bridge/Solver.kt) for 36k training + 600 test decisions (~1 h)
	$(PY) -m genaro.ids --name solve --extra-quiet 24000
	mise exec -- java -jar bridge/bridge.jar dump --hands 3000 --seed 2 --out data/test-decisions-solved.jsonl --solve-ids data/test-solve-ids.txt
	mise exec -- java -jar bridge/bridge.jar dump --hands 30000 --seed 1 --out data/train-decisions-solved.jsonl --solve-ids data/solve-ids.txt

relabel-talk:         # rewrite the lines of talk turns whose move the solver changed (Gemma 4 E4B)
	$(PY) -m genaro.teacher --relabel solver --batch 128 --k 3

solver-play:          # the solver itself in live play vs House Bots (its sanity check as a teacher)
	$(PY) -m genaro.play --mode solver --name solver-3600 --hands 3600 --workers 12

sft-data:             # sft-qwen3-solver (1.7B) and the EV-weighted sft-qwen3-solver-weighted (0.6B)
	$(PY) -m genaro.build --model Qwen/Qwen3-0.6B --label solver --extra-quiet 24000 --out sft-qwen3-solver
	$(PY) -m genaro.build --model Qwen/Qwen3-0.6B --label solver --extra-quiet 24000 --weight-by-gap --out sft-qwen3-solver-weighted

dagger:               # DAgger: the trained model plays 2,400 new hands, the solver labels every state it reaches,
                      # then retrain on the aggregate (DAgger rows count twice). An experiment (no measurable gain); needs out/genaro-poker-0.6b-Q4_K_M.gguf.
	$(PY) -m genaro.play out/genaro-poker-0.6b-Q4_K_M.gguf --name dagger-collect --hands 2400 --workers 12 --seed-offset 100 --collect data/dagger-0.6b.jsonl
	$(PY) -m genaro.build --model Qwen/Qwen3-0.6B --label solver --extra-quiet 24000 --dagger data/dagger-0.6b.jsonl data/dagger-0.6b.jsonl --out sft-qwen3-solver-dagger
	$(PY) -m genaro.sft configs/experiments/genaro-0.6b-dagger.yaml

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
