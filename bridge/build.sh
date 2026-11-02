#!/usr/bin/env bash
# Compiles the bridge together with PokerSI's own (pure Kotlin) engine and prompt code, so the
# prompts, grammars, blur and reply parsing are byte-for-byte what ships in the app.
#   POKERSI=~/Work/PokerSI bridge/build.sh   ->  bridge/bridge.jar
set -euo pipefail
cd "$(dirname "$0")"
POKERSI=${POKERSI:-$HOME/Work/PokerSI}
SRC=$POKERSI/app/src/main/java/com/bradflaugher/pokersi
[[ -d $SRC/engine ]] || { echo "PokerSI sources not found at $POKERSI (set POKERSI=...)"; exit 1; }
git -C "$POKERSI" rev-parse --short HEAD > pokersi.rev
kotlinc -nowarn -include-runtime -d bridge.jar \
  "$SRC"/engine/*.kt \
  "$SRC"/llm/{PokerPrompt,PokerGrammar,Persona,Personas,TalkPlan,LlmReply}.kt \
  Bridge.kt Solver.kt
echo "built bridge.jar against PokerSI $(cat pokersi.rev)"
