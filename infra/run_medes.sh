#!/bin/bash
# Runs every local-backbone configuration of the MEDES 2026 paper on one 24 GB GPU.
#
#   nohup bash infra/run_medes.sh gemma > medes_gemma.log 2>&1 &
#   nohup bash infra/run_medes.sh qwen  > medes_qwen.log  2>&1 &
#   bash infra/run_medes.sh status
#
# For each model: Full, No-Knowledge, No-Execution and tool-forcing under five seeds, and the
# single-call monolithic baseline repeated five times at temperature 0. The served GGUF is
# checked before and after every batch and recorded in each manifest.json; every run resumes
# where it stopped (--resume). Full, No-Execution and tool-forcing read the rule store in
# data/rules (5 seed rules + 84 rules generated from the CIMMYT field guide).

set -e
cd "$(dirname "$0")/.."
source infra/_verify_model.sh

COMPOSE="docker compose -f infra/docker-compose.yml"
SRV=llama-cpp-server
OUT=/app/results/local
SEEDS="${SEEDS:-11 22 33 44 55}"
REPS="${REPS:-1 2 3 4 5}"
PROD=/app/datasets/instances.jsonl
TF=/app/datasets/tool_forcing_instances_n100.jsonl

start_server() {   # $1 = gemma | qwen
  $COMPOSE stop $SRV || true
  if [ "$1" = gemma ]; then
    export LCPP_MODEL_REPO="unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_XL" LCPP_ALIAS=gemma4-local LCPP_CTX_SIZE=8192
  else
    export LCPP_MODEL_REPO="unsloth/Qwen3.6-27B-MTP-GGUF:UD-Q4_K_XL" LCPP_ALIAS=qwen-local LCPP_CTX_SIZE=16384
  fi
  $COMPOSE up -d --force-recreate $SRV
  verify_model "$LCPP_ALIAS"
}

run() {   # $1 = output folder name, then cli args
  local name="$1"; shift
  verify_model "$LCPP_ALIAS"
  $COMPOSE run --rm --no-deps -e TRIAGENT_SEED="${SEED:-}" triagent-runner \
    "$@" --output "$OUT/$name/results.jsonl" --backbone "$LCPP_ALIAS" --provider llamacpp --resume
  verify_model "$LCPP_ALIAS"
}

model_runs() {   # $1 = gemma | qwen
  [ "$(grep -o '<<trigger>>' data/rules/rules.json | wc -l)" -gt 0 ] || { echo "data/rules holds no CIMMYT rules"; exit 1; }
  start_server "$1"
  local tag="$LCPP_ALIAS"
  for SEED in $SEEDS; do
    export SEED
    run "${tag}__full__seed$SEED"         batch --input $PROD --ablation full --skip-prefetch
    run "${tag}__no_knowledge__seed$SEED" batch --input $PROD --ablation no_knowledge --skip-prefetch
    run "${tag}__no_execution__seed$SEED" batch --input $PROD --ablation no_execution
    run "${tag}__toolforcing__seed$SEED"  batch --input $TF --ablation full --skip-prefetch
  done
  SEED=""
  for r in $REPS; do
    run "${tag}__monolithic__rep$r" monolithic --input $PROD
  done
}

case "${1:-}" in
  gemma|qwen) model_runs "$1" ;;
  status)
    wc -l results/local/*/results.jsonl 2>/dev/null | grep -v total || echo "no runs yet"
    grep -h '"served_model_end"' results/local/*/manifest.json 2>/dev/null | sort | uniq -c
    ;;
  *) echo "usage: bash infra/run_medes.sh gemma|qwen|status"; exit 1 ;;
esac
