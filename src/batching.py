import asyncio
import time
from typing import List, Callable
from dataclasses import dataclass, field
from src.config import settings


@dataclass
class InferenceRequest:
    prompt: str
    future: asyncio.Future = field(default_factory=lambda: asyncio.get_event_loop().create_future())
    enqueued_at: float = field(default_factory=time.monotonic)


class DynamicBatcher:
    """
    Hybrid batching: flush when EITHER max_batch_size requests queued
    OR batch_timeout_ms milliseconds have elapsed — whichever comes first.
    """

    def __init__(self, inference_fn: Callable[[List[str]], List[str]]):
        self.inference_fn = inference_fn
        self.max_batch_size = settings.max_batch_size
        self.timeout_s = settings.batch_timeout_ms / 1000.0
        self._queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._task = None

    async def start(self):
        self._running = True
        self._task = asyncio.create_task(self._batch_loop())

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()

    async def infer(self, prompt: str) -> str:
        """Submit a prompt and wait for the batch result."""
        req = InferenceRequest(prompt=prompt)
        await self._queue.put(req)
        return await req.future

    async def _batch_loop(self):
        while self._running:
            batch: List[InferenceRequest] = []
            deadline = time.monotonic() + self.timeout_s

            while len(batch) < self.max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    req = await asyncio.wait_for(
                        self._queue.get(), timeout=remaining
                    )
                    batch.append(req)
                except asyncio.TimeoutError:
                    break

            if not batch:
                continue

            try:
                prompts = [r.prompt for r in batch]
                loop = asyncio.get_event_loop()
                results = await loop.run_in_executor(
                    None, self.inference_fn, prompts
                )
                for req, result in zip(batch, results):
                    if not req.future.done():
                        req.future.set_result(result)
            except Exception as exc:
                for req in batch:
                    if not req.future.done():
                        req.future.set_exception(exc)

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()
