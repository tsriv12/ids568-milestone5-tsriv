import asyncio
import time
import psutil
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from src.config import settings
from src.batching import DynamicBatcher
from src.caching import get_cache, make_cache_key

# Global state
model_pipeline = None
batcher: Optional[DynamicBatcher] = None
cache = get_cache()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_pipeline, batcher
    print(f"Loading model: {settings.model_name} on {settings.device}")
    model_pipeline = pipeline(
        "text-generation",
        model=settings.model_name,
        device=-1,
        max_new_tokens=settings.max_new_tokens,
    )
    print("Model loaded.")

    def _run_batch(prompts: list) -> list:
        outputs = model_pipeline(prompts, batch_size=len(prompts))
        return [o[0]["generated_text"] for o in outputs]

    batcher = DynamicBatcher(inference_fn=_run_batch)
    await batcher.start()
    print("Batcher started.")
    yield
    await batcher.stop()
    print("Server shutdown.")


app = FastAPI(title="MLOps Milestone 5 Inference Server", lifespan=lifespan)


class InferenceRequest(BaseModel):
    prompt: str
    use_cache: bool = True


class InferenceResponse(BaseModel):
    result: str
    cache_hit: bool
    latency_ms: float
    batch_queue_depth: int


@app.post("/infer", response_model=InferenceResponse)
async def infer(req: InferenceRequest):
    start = time.monotonic()
    key = make_cache_key(req.prompt, settings.model_name, settings.max_new_tokens)

    if req.use_cache:
        cached = await cache.get(key)
        if cached:
            return InferenceResponse(
                result=cached,
                cache_hit=True,
                latency_ms=(time.monotonic() - start) * 1000,
                batch_queue_depth=batcher.queue_depth,
            )

    result = await batcher.infer(req.prompt)

    if req.use_cache:
        await cache.set(key, result)

    return InferenceResponse(
        result=result,
        cache_hit=False,
        latency_ms=(time.monotonic() - start) * 1000,
        batch_queue_depth=batcher.queue_depth,
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model": settings.model_name}


@app.get("/metrics")
async def metrics():
    cache_stats = await cache.stats()
    return {
        "cache": cache_stats,
        "queue_depth": batcher.queue_depth,
        "cpu_pct": psutil.cpu_percent(),
        "ram_used_gb": round(psutil.virtual_memory().used / 1e9, 2),
    }


@app.delete("/cache")
async def clear_cache():
    await cache.flush()
    return {"status": "cache cleared"}
