# Parrot baselines quickstart

The commands below condense `baselines/ParrotServe/docs/get_started` and `baselines/ParrotServe/examples` into a runbook for the Parrot baselines in this folder. Run them from the repo root unless noted.

## 0) Install ParrotServe (one time)
```bash
cd baselines/ParrotServe
pip install -r requirements.txt
pip install -e .
# Optional extras from docs/get_started/installation.md
cd 3rdparty/vllm && pip install -e . && cd ../..
cd 3rdparty/FastChat && pip install -e ".[model_worker,webui]" && cd ../..
cd 3rdparty/langchain/libs/langchain && pip install -e . && cd ../../../..
```

## 1) Start Parrot core and engines (docs/get_started/launch_server.md)
```bash
# ServeCore
mkdir -p ./baselines/logs/
SIMULATE_NETWORK_LATENCY_PRT=0 python -m parrot.serve.http_server \
  --config_path baselines/ParrotServe/sample_configs/core/localhost_serve_core.json 

# Engines (one GPU each; configs under baselines/parrot_configs/*.json)
CUDA_VISIBLE_DEVICES=0 python -m parrot.engine.http_server \
  --config_path baselines/parrot/model_Gptoss20b.json 

CUDA_VISIBLE_DEVICES=1 python -m parrot.engine.http_server \
  --config_path baselines/parrot/model_Qwen14b.json 

CUDA_VISIBLE_DEVICES=2 python -m parrot.engine.http_server \
  --config_path baselines/parrot/model_Qwen32b.json 

# Copy/modify the engine config so engine_name/model match the names passed via *_MODEL_* below
# (e.g., openai/gpt-oss-20b, Qwen/Qwen3-14B, Qwen/Qwen3-32B).
# For Azure/OpenAI, fill sample_configs/engine/openai-example-config.json and launch similarly.
```

### If you prefer running the models via vLLM (OpenAI-compatible HTTP)
Make sure the ports line up with your *_MODEL_* values and Parrot engine configs (or set PARROT_OPENAI_BASE).
```bash
CUDA_VISIBLE_DEVICES=0 vllm serve openai/gpt-oss-20b --host 0.0.0.0 --port 9101
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-14B    --host 0.0.0.0 --port 9102
CUDA_VISIBLE_DEVICES=2 vllm serve Qwen/Qwen3-32B    --host 0.0.0.0 --port 9103
```

## 2) Start the shared DB worker (for IMDb/FineWiki SQL)
```bash
# Defaults: PORT=9104, HALO_PG_PORT=4032, HALO_PG_DBNAME=imdb
PORT=9104 bash baselines/run_db_worker_unicorn.sh
# Export HALO_PG_HOST / HALO_PG_USER / HALO_PG_PASSWORD as needed before running.
```

## 3) Optional sanity check from docs/examples
```bash
cd baselines/ParrotServe
python examples/hello_world.py
python examples/write_recommendation_letter.py
cd ../..
```

## 4) Run the Parrot baselines in this folder
Make sure `*_MODEL_*` values match the engine names already registered to ServeCore.

```bash
# FineWiki long-chain
PARROT_CORE_HTTP=http://localhost:9000 \
HALO_DB_WORKER_URL=http://localhost:9104 \
FINEWIKI_MODEL_A=openai/gpt-oss-20b \
FINEWIKI_MODEL_B=Qwen/Qwen3-32B \
FINEWIKI_MODEL_C=Qwen/Qwen3-14B \
python baselines/parrot_configs/finewiki_long_chain_parrot.py

# FineWiki bridges
PARROT_CORE_HTTP=http://localhost:9000 \
HALO_DB_WORKER_URL=http://localhost:9104 \
FINEWIKI_MODEL_A=openai/gpt-oss-20b \
FINEWIKI_MODEL_B=Qwen/Qwen3-32B \
FINEWIKI_MODEL_C=Qwen/Qwen3-14B \
python baselines/parrot_configs/finewiki_bridges_parrot.py

# IMDb triple-chain
PARROT_CORE_HTTP=http://localhost:9000 \
HALO_DB_WORKER_URL=http://localhost:9104 \
IMDB_MODEL_A=openai/gpt-oss-20b \
IMDB_MODEL_B=Qwen/Qwen3-14B \
IMDB_MODEL_C=Qwen/Qwen3-32B \
python baselines/parrot_configs/imdb_triple_chain_parrot.py

# IMDb diamond
PARROT_CORE_HTTP=http://localhost:9000 \
HALO_DB_WORKER_URL=http://localhost:9104 \
IMDB_MODEL_A=openai/gpt-oss-20b \
IMDB_MODEL_B=Qwen/Qwen3-14B \
IMDB_MODEL_C=Qwen/Qwen3-32B \
python baselines/parrot_configs/imdb_diamond_parrot.py
```

Notes:
- `HALO_DB_TIMEOUT` controls DB worker timeout (default 1200s).
- Run each of the core/engine/baseline commands in separate terminals if you want live logs, as suggested in docs/get_started/run_example.md.
