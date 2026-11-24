#!/bin/bash
# Confirms that the llama.cpp server has the expected model loaded before a batch runs.
# The alias (-a) is only a label, so the GGUF path reported by /props is checked as well.
# Polls until the server answers: loading a 15-30 GB GGUF into VRAM can take minutes.
#
# Usage: source this file, then call `verify_model <alias>`.
# Optional: VERIFY_MODEL_TIMEOUT_S (default 600), VERIFY_MODEL_POLL_S (default 10).

verify_model() {
  local alias="$1"
  local host="${LCPP_HOST:-http://localhost:8080}"
  local timeout_s="${VERIFY_MODEL_TIMEOUT_S:-600}"
  local poll_s="${VERIFY_MODEL_POLL_S:-10}"
  local deadline=$(( $(date +%s) + timeout_s ))
  local models expect path loaded

  case "$alias" in
    gemma4-local) expect="gemma-4" ;;
    qwen-local)   expect="qwen3" ;;
    *)            expect="" ;;
  esac

  while [ "$(date +%s)" -lt "$deadline" ]; do
    if models=$(curl -fsS -m 5 "${host}/v1/models" 2>/dev/null); then
      if ! echo "$models" | grep -q "\"id\":\"${alias}\""; then
        loaded=$(echo "$models" | grep -o '"id":"[^"]*"' | tr '\n' ' ')
        echo "[verify_model] ERROR: expected '${alias}', server reports: ${loaded}"
        exit 1
      fi
      path=$(curl -fsS -m 5 "${host}/props" | grep -o '"model_path":"[^"]*"')
      if [ -n "$expect" ] && ! echo "$path" | grep -qi "$expect"; then
        echo "[verify_model] ERROR: alias '${alias}' is up but the loaded file is: ${path:-not reported}"
        exit 1
      fi
      echo "[verify_model] ok: '${alias}' is loaded (${path})"
      return 0
    fi
    sleep "$poll_s"
  done

  echo "[verify_model] ERROR: ${host}/v1/models not reachable within ${timeout_s}s"
  exit 1
}
