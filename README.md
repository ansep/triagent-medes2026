# TriAgent: code, instances and runs (MEDES 2026)

Code, instances and all runs behind the MEDES 2026 paper *TriAgent: A three-layer LLM-based agentic framework for web-based process automation* (R. Grati, A. Sepielli, K. Boukadi, E. Arba, F. Monti, M. Mecella).

TriAgent decomposes cognitive process automation into three collaborating agents, implemented with LangGraph:

- **Knowledge Agent**: generates structured process rules from documents (offline) and retrieves the closest rule for a case from a FAISS store (all-MiniLM-L6-v2 embeddings, threshold 0.75), generating one on the fly when no rule is close enough;
- **Reasoning Agent**: follows the rule with a ReAct loop and decides each process gateway, flagging low-confidence decisions for human review;
- **Execution Agent**: calls external tools (OpenWeatherMap weather, EPPO pest database, SMS), retrying twice before escalating.

The evaluation covers five crop-disease processes (wheat brown rust, tomato late blight, olive fruit fly, wheat fusarium, citrus greening) with one to four gateways each.

## What is in this repository

| Path | Content |
|---|---|
| `backend/` | TriAgent (agents, orchestrator, CLI used for all runs) |
| `data/rules/` | FAISS rule store used at run time: 5 manually curated seed rules (one per process) and 84 rules generated offline by DeepSeek-V3 from the CIMMYT field guide |
| `datasets/instances.jsonl` | production dataset: 100 instances, 5 processes × 20 variants (SHA-256 in `instances.jsonl.sha256`) |
| `datasets/ground_truth.jsonl` | expected diagnosis, treatment, dose, gateway decisions, tool calls and escalation per instance |
| `datasets/tool_forcing_instances_n100.jsonl`, `ground_truth_toolforcing_n100.jsonl` | tool-forcing dataset: weather removed from the descriptions, so gateways need a live weather API call |
| `datasets/seed_rules.jsonl` | the five seed rules, in the format produced by the rule generator |
| `experiments/evaluate.py` | per-instance scoring of the six dimensions |
| `experiments/analyze_medes.py` | Tables 1 and 2 of the paper: mean and 95% CI over seeds, agent errors, latency, paired Full − No-Knowledge difference |
| `experiments/build_dataset.py`, `generate_toolforcing.py` | how the two datasets were generated |
| `experiments/load_fao_docs.py` | offline rule generation from documents |
| `infra/` | Docker runner, llama.cpp server pinned by digest, `run_medes.sh` |
| `results/local/` | Gemma4-26B and Qwen3.6-27B: Full, No-Knowledge, No-Execution and tool-forcing under seeds 11, 22, 33, 44, 55; single-call baseline repeated five times at temperature 0 (50 runs) |
| `results/deepseek/` | DeepSeek-V3 via API: one run per configuration (5 runs) |
| `results/analysis_output.txt` | output of `analyze_medes.py` on `results/` |

Each run folder contains `results.jsonl` (one line per instance, with the full reasoning trace and the rule used) and `manifest.json` (dataset hash, seed and, for the local models, the GGUF file actually served at the start and end of the batch).

## Setup of the paper

- **Local models:** `unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_XL` (snapshot `c099eb48…`) and `unsloth/Qwen3.6-27B-MTP-GGUF:UD-Q4_K_XL` (snapshot `5cb35eb3…`), served one at a time by llama.cpp (`server-cuda` image of 9 June 2026, pinned by digest in `infra/docker-compose.yml`) on a single 24 GB GPU, reasoning off.
- **Sampling:** the Reasoning Agent uses each model's recommended settings (Gemma: T = 1.0; Qwen: T = 0.7), with the seed set per run through `TRIAGENT_SEED`; the Knowledge Agent and the baseline use greedy decoding. DeepSeek-V3 is run once per configuration, greedy, via the DeepSeek API.
- **Configurations:** Full; No-Knowledge (rule bypassed, raw description passed to the Reasoning Agent); No-Execution (tools disabled); tool-forcing (Full on the tool-forcing dataset); monolithic single-call baseline (diagnosis and treatment in two lines).

## Reproducing the tables

The analysis needs only the Python standard library:

```bash
python3 experiments/analyze_medes.py results
```

The output matches `results/analysis_output.txt` and Tables 1 and 2 of the paper.

## Re-running the experiments

```bash
cp backend/.env.example backend/.env       # add an OpenWeatherMap key (and a DeepSeek key for the API runs)
docker compose -f infra/docker-compose.yml build
nohup bash infra/run_medes.sh gemma > medes_gemma.log 2>&1 &
nohup bash infra/run_medes.sh qwen  > medes_qwen.log  2>&1 &
bash infra/run_medes.sh status
```

Every run uses `--resume`: rows that failed for infrastructure reasons are retried, successful rows are kept. The served model is checked before and after every batch (`infra/_verify_model.sh`).

## Notes for interpreting the runs

- **Agent errors** (for example, malformed structured output) count as incorrect on every dimension; their number is printed per configuration.
- **Rule store.** 38 of the 100 production instances retrieve a stored rule above the threshold: 34 a seed rule and 4 (all of P1) a rule generated from the CIMMYT guide. The other instances receive a rule generated on the fly from their description. The paper reports the gateway gain separately for these groups.
- **Gateway check.** The trace does not label decisions by gateway, so the check passes when every expected outcome appears among the decisions in the trace: an upper bound on per-gateway correctness.
- **No-Knowledge and baseline runs** do not read the rule store; they come from a separate batch on the same server image, model snapshots and seeds.
- **Weather.** The Execution Agent queries live weather, so tool observations differ across runs and days.
- **Documents.** The CIMMYT field guide (*Wheat Diseases and Pests: A Guide for Field Identification*, Prescott et al., 1986) is not redistributed here; it is available at http://hdl.handle.net/10883/3215.

## License

MIT (see `LICENSE`).
