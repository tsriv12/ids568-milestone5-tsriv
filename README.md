# ids568-milestone5-tsriv

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Run Server
```bash
uvicorn src.server:app --host 0.0.0.0 --port 8000
```

## Run Benchmarks
```bash
python3 benchmarks/run_benchmarks.py --max-concurrency 15
```

## Configuration
| Variable | Default | Description |
|---|---|---|
| MLOPS_MAX_BATCH_SIZE | 8 | Max requests per batch |
| MLOPS_BATCH_TIMEOUT_MS | 50 | Batch flush timeout (ms) |
| MLOPS_CACHE_TTL_SECONDS | 300 | Cache entry TTL |
| MLOPS_CACHE_MAX_ENTRIES | 1000 | Max cached entries |
| MLOPS_MODEL_NAME | facebook/opt-125m | HuggingFace model |
