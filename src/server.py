import os
os.environ.setdefault("USE_TF", "0")

import asyncio
import time
import psutil
from contextlib import asynccontextmanager
from typing import Optional, Any

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

from src.config import settings
from src.batching import DynamicBatcher
from src.caching import get_cache, make_cache_key

try:
    import torch
except Exception:
    torch = None

try:
    import GPUtil
except Exception:
    GPUtil = None


model_pipeline = None
batcher: Optional[DynamicBatcher] = None
cache = get_cache()


def _resolve_pipeline_device() -> int:
    raw_device = str(settings.device).strip().lower()

    if raw_device == "cpu":
        return -1

    if raw_device in {"cuda", "gpu"}:
        if torch is not None and torch.cuda.is_available():
            return 0
        print("WARNING: GPU requested but CUDA is not available. Falling back to CPU.")
        return -1

    if raw_device.startswith("cuda:"):
        try:
            gpu_index = int(raw_device.split(":", 1)[1])
            if torch is not None and torch.cuda.is_available():
                if gpu_index < torch.cuda.device_count():
                    return gpu_index
            print(f"WARNING: Requested GPU index {gpu_index} is not available. Falling back to CPU.")
            return -1
        except ValueError:
            print(f"WARNING: Invalid device setting '{settings.device}'. Falling back to CPU.")
            return -1

    print(f"WARNING: Unrecognized device setting '{settings.device}'. Falling back to CPU.")
    return -1


def _collect_gpu_metrics() -> dict[str, Any]:
    gpu_metrics: dict[str, Any] = {
        "gpu_available": False,
        "gpu_count": 0,
        "gpu_util_pct": None,
        "gpu_mem_used_mb": None,
        "gpu_mem_total_mb": None,
        "gpu_mem_util_pct": None,
    }

    if torch is None or not torch.cuda.is_available():
        return gpu_metrics

    gpu_metrics["gpu_available"] = True
    gpu_metrics["gpu_count"] = torch.cuda.device_count()

    if GPUtil is not None:
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                gpu_metrics["gpu_util_pct"] = round(gpu.load * 100, 2)
                gpu_metrics["gpu_mem_used_mb"] = round(float(gpu.memoryUsed), 2)
                gpu_metrics["gpu_mem_total_mb"] = round(float(gpu.memoryTotal), 2)
                if gpu.memoryTotal:
                    gpu_metrics["gpu_mem_util_pct"] = round(
                        (float(gpu.memoryUsed) / float(gpu.memoryTotal)) * 100, 2
                    )
                return gpu_metrics
        except Exception:
            pass

    try:
        current_idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(current_idx)
        total_mb = props.total_memory / (1024 * 1024)
        used_mb = torch.cuda.memory_allocated(current_idx) / (1024 * 1024)

        gpu_metrics["gpu_mem_used_mb"] = round(used_mb, 2)
        gpu_metrics["gpu_mem_total_mb"] = round(total_mb, 2)
        if total_mb:
            gpu_metrics["gpu_mem_util_pct"] = round((used_mb / total_mb) * 100, 2)
    except Exception:
        pass

    return gpu_metrics


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_pipeline, batcher

    pipeline_device = _resolve_pipeline_device()
    device_label = "cpu" if pipeline_device == -1 else f"cuda:{pipeline_device}"
    print(f"Loading model: {settings.model_name} on {device_label}")

    model_pipeline = pipeline(
        "text-generation",
        model=settings.model_name,
        device=pipeline_device,
        max_new_tokens=settings.max_new_tokens,
    )
    print("Model loaded.")

    def _run_batch(prompts: list[str]) -> list[str]:
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
                batch_queue_depth=batcher.queue_depth if batcher else 0,
            )

    result = await batcher.infer(req.prompt)

    if req.use_cache:
        await cache.set(key, result)

    return InferenceResponse(
        result=result,
        cache_hit=False,
        latency_ms=(time.monotonic() - start) * 1000,
        batch_queue_depth=batcher.queue_depth if batcher else 0,
    )


@app.get("/health")
async def health():
    pipeline_device = _resolve_pipeline_device()
    return {
        "status": "ok",
        "model": settings.model_name,
        "configured_device": settings.device,
        "runtime_device": "cpu" if pipeline_device == -1 else f"cuda:{pipeline_device}",
    }


@app.get("/metrics")
async def metrics():
    cache_stats = await cache.stats()
    gpu_stats = _collect_gpu_metrics()

    return {
        "cache": cache_stats,
        "queue_depth": batcher.queue_depth if batcher else 0,
        "cpu_pct": psutil.cpu_percent(),
        "ram_used_gb": round(psutil.virtual_memory().used / 1e9, 2),
        **gpu_stats,
    }


@app.delete("/cache")
async def clear_cache():
    await cache.flush()
    return {"status": "cache cleared"}
