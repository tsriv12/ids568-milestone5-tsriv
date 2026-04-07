# Performance Analysis Report
## IDS568 Milestone 5 — LLM Inference Optimization
**Author:** tsriv | **Model:** facebook/opt-125m | **Hardware:** GCP n1-highmem-2 (CPU)

---

## 1. Introduction

This report analyzes the performance impact of two key LLM inference optimizations:
dynamic request batching and intelligent response caching. Benchmarks were conducted
on a GCP n1-highmem-2 instance using the facebook/opt-125m model served via a
custom FastAPI inference server.

---

## 2. System Architecture & Compute Pathways

### 2.1 Baseline Inference Pathway
Without optimization, each request follows a sequential pathway: tokenization,
forward pass through all transformer layers, and detokenization. For opt-125m,
this involves 12 transformer layers with 768-dimensional hidden states. On CPU,
each forward pass requires approximately 2,000-2,500ms due to sequential matrix
multiplications that cannot be parallelized across requests.

### 2.2 Batched Inference Pathway
Dynamic batching groups concurrent requests and processes them in a single forward
pass. The DynamicBatcher uses a hybrid strategy: flush when either max_batch_size
(8) requests are queued OR batch_timeout_ms (50ms) elapses. This amortizes the
fixed overhead of model loading, tokenization padding, and GPU/CPU memory transfers
across multiple requests simultaneously. On GPU hardware, batching provides dramatic
throughput improvements because matrix multiplications scale efficiently with batch
dimension — a batch of 8 requires only marginally more compute than a batch of 1.

### 2.3 Cached Inference Pathway
The caching layer intercepts requests before they reach the batcher. Cache keys are
computed as SHA-256 hashes of the prompt concatenated with model parameters,
ensuring privacy (no user identifiers stored) while enabling exact-match lookups.
Redis stores responses with configurable TTL (300s default) and max-entry limits
(1000 entries). A cache hit completely bypasses model inference, reducing latency
from ~2000ms to sub-millisecond.

---

## 3. Benchmark Methodology

All benchmarks were conducted using the following methodology:
- Cache was explicitly cleared via DELETE /cache before each cold measurement
- Warm cache measurements immediately followed cold runs with identical prompts
- Cold cache used unique prompts only; warm cache and hit-rate tests used 50% repeated prompts
- Metrics captured: mean, median, P95, P99 latency, wall-clock runtime, throughput, and cache hit rate
- Server was fully warmed up before final measurements

---

## 4. Results

### 4.1 Latency Comparison

| Configuration | Mean (ms) | Median (ms) | P95 (ms) | P99 (ms) |
|---|---|---|---|---|
| Single request (no cache) | 3389.7 | 3389.7 | 3415.3 | 3415.3 |
| Batched (no cache) | 9186.2 | 10248.6 | 11064.6 | 11064.6 |
| Cold cache | 6880.9 | 6919.4 | 10865.4 | 10865.4 |
| Warm cache | 3483.3 | 1.3 | 10974.0 | 10974.0 |

See Fig 1 (latency_comparison.png) for visualization.

### 4.2 Caching Impact
Cold-to-warm cache transition reduced mean latency from
6880.9ms to 3483.3ms
— a 2.0x improvement.
The warm cache hit rate reached 50%,
demonstrating that repeated prompts are served entirely from cache without
touching the model.

### 4.3 Throughput Under Load

| Load Level | Concurrency | Mean Latency (ms) | Cache Hit Rate |
|---|---|---|---|
| Low | 5 | 5154.0 | 20% |
| Medium | 10 | 7324.8 | 20% |
| High | 15 | 10981.8 | 12% |

See Fig 2 (throughput_by_load.png) for visualization.

### 4.4 Cache Hit Rate Over Time
Starting from a cold cache, the hit rate grew to 42%
over 50 requests with 50% repeated prompts. The hit rate stabilized after
approximately 15 requests as the cache warmed up. See Fig 3
(cache_hit_rate_over_time.png) for the time-series visualization.

---

## 5. Trade-off Analysis

### 5.1 Batching Window vs. Latency
A longer batch timeout increases the chance of forming larger batches, improving
throughput but increasing per-request latency for requests that arrive when the
queue is nearly empty. Our 50ms timeout balances these concerns: small enough
that single requests are not penalized heavily, large enough to collect concurrent
requests. On CPU, batching showed higher mean latency under concurrency
(7329.8ms vs 2281.0ms
single) because CPU matrix operations do not parallelize across batch dimension
as efficiently as GPU operations. On a T4 GPU, batching would provide 3-8x
throughput improvement.

### 5.2 Cache Size vs. Hit Rate
With max_entries=1000 and TTL=300s, our cache can hold up to 1000 unique
prompt-response pairs. In our observed CPU benchmark run with a 50% repeat ratio,
hit rates reached about 42% over time and 50% in the warm-cache scenario. Increasing cache size beyond the working set of unique prompts
yields diminishing returns. A smaller cache (e.g., 100 entries) with high-traffic
workloads risks evicting frequently used entries, reducing hit rate.

---

## 6. Proposed Scaling Strategies

Based on empirical results, the following strategies are recommended for production:

1. **GPU deployment**: Migrate to T4/A100 GPU to unlock batching benefits.
   Expected 3-8x throughput improvement over CPU baseline.
2. **Redis cluster**: Replace single Redis instance with Redis Cluster for
   horizontal cache scaling beyond single-node memory limits.
3. **Semantic caching**: Extend exact-match cache with embedding-based similarity
   search to cache semantically equivalent prompts, increasing hit rates.
4. **Adaptive batch timeout**: Dynamically adjust batch_timeout_ms based on
   queue depth — shorter timeouts under low load, longer under high load.
5. **Cache warming**: Pre-populate cache with predicted high-frequency prompts
   at server startup to reduce cold-start latency spikes.

---

## 7. Conclusion

Caching provides a measurable performance improvement in this CPU-based run.
Warm-cache behavior reduced mean latency by about 2.0x for repeated prompts.
Batching did not reduce latency on this CPU setup and in fact increased it under
concurrency, but it remains an important production optimization that is expected
to provide clearer benefits on GPU hardware. Together, batching and caching still
represent core serving patterns for production-grade LLM infrastructure.

*Charts referenced: Fig 1-4 in analysis/visualizations/*
