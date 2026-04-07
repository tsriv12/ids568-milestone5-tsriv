import asyncio
import aiohttp
import json
import time
import argparse
import statistics
from pathlib import Path
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.load_generator import run_load

RESULTS_DIR = Path("benchmarks/results")
BASE_URL = "http://localhost:8000"


async def clear_cache():
    async with aiohttp.ClientSession() as s:
        await s.delete(f"{BASE_URL}/cache")


async def get_metrics():
    async with aiohttp.ClientSession() as s:
        async with s.get(f"{BASE_URL}/metrics") as r:
            return await r.json()


def summarize(results: list, wall_time_s: float | None = None) -> dict:
    latencies = [r["client_latency_ms"] for r in results if "error" not in r]
    if not latencies:
        return {"error": "all requests failed"}
    sorted_l = sorted(latencies)
    hits = sum(1 for r in results if r.get("cache_hit"))
    n = len(latencies)
    return {
        "n": n,
        "cache_hits": hits,
        "hit_rate": round(hits / len(results), 3),
        "mean_ms": round(statistics.mean(latencies), 2),
        "median_ms": round(statistics.median(latencies), 2),
        "p95_ms": round(sorted_l[min(int(n * 0.95), n - 1)], 2),
        "p99_ms": round(sorted_l[min(int(n * 0.99), n - 1)], 2),
        "min_ms": round(sorted_l[0], 2),
        "max_ms": round(sorted_l[-1], 2),
        "wall_time_s": round(wall_time_s, 3) if wall_time_s is not None else None,
        "throughput_rps": round(n / wall_time_s, 2) if wall_time_s and wall_time_s > 0 else None,
    }


async def run_all(args):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    print("\n[1/5] Single-request baseline (no cache, concurrency=1)...")
    await clear_cache()
    t0 = time.perf_counter()
    r = await run_load(f"{BASE_URL}/infer", 1, 10, 0.0, use_cache=False)
    wall_time_s = time.perf_counter() - t0
    all_results["single_request_no_cache"] = summarize(r, wall_time_s)
    all_results["single_request_no_cache"]["metrics"] = await get_metrics()
    print(f"      Mean: {all_results['single_request_no_cache']['mean_ms']:.1f}ms")

    print("\n[2/5] Batched requests (no cache, high concurrency)...")
    await clear_cache()
    t0 = time.perf_counter()
    r = await run_load(f"{BASE_URL}/infer", args.max_concurrency, 20, 0.0, use_cache=False)
    wall_time_s = time.perf_counter() - t0
    all_results["batched_no_cache"] = summarize(r, wall_time_s)
    all_results["batched_no_cache"]["metrics"] = await get_metrics()
    print(f"      Mean: {all_results['batched_no_cache']['mean_ms']:.1f}ms")

    print("\n[3/5] Cold cache vs warm cache (cold=unique prompts, warm=50% repeats)...")
    await clear_cache()
    t0 = time.perf_counter()
    r_cold = await run_load(f"{BASE_URL}/infer", 5, 20, 0.0, use_cache=True)
    wall_time_s = time.perf_counter() - t0
    all_results["cold_cache"] = summarize(r_cold, wall_time_s)
    all_results["cold_cache"]["metrics"] = await get_metrics()
    print(f"      Cold mean: {all_results['cold_cache']['mean_ms']:.1f}ms")

    t0 = time.perf_counter()
    r_warm = await run_load(f"{BASE_URL}/infer", 5, 20, 0.5, use_cache=True)
    wall_time_s = time.perf_counter() - t0
    all_results["warm_cache"] = summarize(r_warm, wall_time_s)
    all_results["warm_cache"]["metrics"] = await get_metrics()
    print(f"      Warm mean: {all_results['warm_cache']['mean_ms']:.1f}ms")

    print("\n[4/5] Throughput at multiple load levels...")
    for label, conc, n in [("low_10rps", 5, 15), ("med_50rps", 10, 20), ("high_100rps", 15, 25)]:
        await clear_cache()
        t0 = time.perf_counter()
        r = await run_load(f"{BASE_URL}/infer", conc, n, 0.3, use_cache=True)
        wall_time_s = time.perf_counter() - t0
        all_results[f"throughput_{label}"] = summarize(r, wall_time_s)
        all_results[f"throughput_{label}"]["metrics"] = await get_metrics()
        print(f"      {label}: mean={all_results[f'throughput_{label}']['mean_ms']:.1f}ms, hit_rate={all_results[f'throughput_{label}']['hit_rate']:.2f}")

    print("\n[5/5] Cache hit rate over time (50 requests, 50% repeat)...")
    await clear_cache()
    t0 = time.perf_counter()
    r = await run_load(f"{BASE_URL}/infer", 5, 50, 0.5, use_cache=True)
    wall_time_s = time.perf_counter() - t0
    with open(RESULTS_DIR / "hit_rate_over_time.json", "w") as f:
        json.dump(r, f, indent=2)
    all_results["hit_rate_series"] = summarize(r, wall_time_s)
    all_results["hit_rate_series"]["metrics"] = await get_metrics()
    print(f"      Final hit rate: {all_results['hit_rate_series']['hit_rate']:.2f}")

    metrics = await get_metrics()
    all_results["final_metrics"] = metrics

    timestamp = int(time.time())
    out_path = RESULTS_DIR / f"benchmark_results_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✓ Results saved to {out_path}")
    print("\n=== SUMMARY ===")
    for k, v in all_results.items():
        if k != "final_metrics" and isinstance(v, dict) and "mean_ms" in v:
            print(f"  {k:35s} mean={v['mean_ms']:8.1f}ms  hits={v['cache_hits']}/{v['n']}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full benchmark suite for Milestone 5")
    parser.add_argument("--max-concurrency", type=int, default=15,
                        help="Max concurrent requests for batching tests")
    args = parser.parse_args()
    asyncio.run(run_all(args))
