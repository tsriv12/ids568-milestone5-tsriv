import asyncio
import aiohttp
import time
import random
import argparse
import statistics
from typing import List

SAMPLE_PROMPTS = [
    "Explain machine learning in simple terms",
    "What is the capital of France?",
    "Write a short poem about technology",
    "The best way to learn programming is",
    "Describe the water cycle",
    "What causes climate change?",
    "How do neural networks work?",
    "The history of the internet began",
    "Explain quantum computing briefly",
    "What is transfer learning?",
]

REPEATED_PROMPT = "What is machine learning?"


async def send_request(session: aiohttp.ClientSession, url: str,
                       prompt: str, use_cache: bool) -> dict:
    start = time.monotonic()
    try:
        async with session.post(
            url,
            json={"prompt": prompt, "use_cache": use_cache},
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            data = await resp.json()
            data["client_latency_ms"] = (time.monotonic() - start) * 1000
            return data
    except Exception as e:
        return {"error": str(e), "client_latency_ms": (time.monotonic() - start) * 1000}


async def run_load(url: str, concurrency: int, total_requests: int,
                   repeat_ratio: float, use_cache: bool) -> List[dict]:
    results = []
    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        sem = asyncio.Semaphore(concurrency)

        async def bounded_request(i: int):
            async with sem:
                if random.random() < repeat_ratio:
                    prompt = REPEATED_PROMPT
                else:
                    prompt = random.choice(SAMPLE_PROMPTS)
                result = await send_request(session, url, prompt, use_cache)
                result["request_id"] = i
                return result

        tasks = [bounded_request(i) for i in range(total_requests)]
        results = await asyncio.gather(*tasks)

    return list(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load generator for inference server")
    parser.add_argument("--url", default="http://localhost:8000/infer")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--repeat-ratio", type=float, default=0.3)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    results = asyncio.run(run_load(
        url=args.url,
        concurrency=args.concurrency,
        total_requests=args.requests,
        repeat_ratio=args.repeat_ratio,
        use_cache=not args.no_cache,
    ))

    latencies = [r["client_latency_ms"] for r in results if "error" not in r]
    hits = sum(1 for r in results if r.get("cache_hit"))
    errors = sum(1 for r in results if "error" in r)

    print(f"Completed : {len(latencies)}/{args.requests} successful, {errors} errors")
    print(f"Cache hits: {hits} ({hits/len(results)*100:.1f}%)")
    if latencies:
        print(f"Mean      : {statistics.mean(latencies):.1f}ms")
        print(f"Median    : {statistics.median(latencies):.1f}ms")
        print(f"P95       : {sorted(latencies)[int(len(latencies)*0.95)]:.1f}ms")
