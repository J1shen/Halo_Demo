### If you prefer running the models via vLLM (OpenAI-compatible HTTP)
Make sure the ports line up with your *_MODEL_* values and Parrot engine configs (or set PARROT_OPENAI_BASE).
```bash
CUDA_VISIBLE_DEVICES=0 vllm serve openai/gpt-oss-20b --host 0.0.0.0 --port 9101

CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-14B    --host 0.0.0.0 --port 9102

CUDA_VISIBLE_DEVICES=2 vllm serve Qwen/Qwen3-32B    --host 0.0.0.0 --port 9103

sh ./baselines/run_db_worker_unicorn.sh
```