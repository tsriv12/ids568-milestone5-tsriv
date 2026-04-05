# Governance Memo: LLM Inference Caching
## IDS568 Milestone 5
**Author:** tsriv | **Date:** April 2026

---

## 1. Privacy Considerations for Cached Inputs/Outputs

The inference server caches both prompts (inputs) and generated responses (outputs).
This introduces privacy risks if not handled carefully. Our implementation addresses
these through privacy-preserving cache key design: keys are computed as SHA-256
hashes of the prompt concatenated with model parameters only. No user identifiers,
session tokens, IP addresses, or any personally identifiable information (PII) are
stored in cache keys or values. This ensures that even if the cache store is
compromised, individual users cannot be identified from cached entries.

However, the cached response itself may contain sensitive information if the
original prompt contained PII (e.g., a prompt asking about a specific person).
Organizations must implement prompt filtering before caching to detect and exclude
sensitive content from the cache entirely.

## 2. Data Retention and Expiration Policies

Our cache implements configurable TTL (time-to-live) with a default of 300 seconds
(5 minutes) and a maximum of 1000 entries. These defaults reflect a balance between
performance and data minimization principles. For production deployments, TTL values
should be determined by the sensitivity of the data:

- **Public/non-sensitive prompts:** TTL of 1-24 hours is acceptable
- **Internal business prompts:** TTL of 15-60 minutes recommended
- **Any prompts involving personal data:** TTL should not exceed session duration,
  or caching should be disabled entirely for these request types

Cache entries must be automatically expired and not retained beyond their TTL.
The Redis backend enforces this via native key expiration. Organizations must
also implement a mechanism to honor user deletion requests (right to erasure)
by providing cache invalidation endpoints, as implemented in our DELETE /cache
endpoint.

## 3. Potential Misuse Scenarios

**Cache Poisoning:** A malicious actor could craft prompts designed to populate
the cache with harmful or misleading responses. Subsequent users with similar
prompts would receive the poisoned cached response. Mitigation: implement
response validation before caching and rate-limit cache writes per IP.

**Inference Extraction:** Repeated querying with systematic prompt variations
could allow an adversary to extract information about what prompts are cached
(via timing side-channels — cache hits return faster than misses). This could
reveal usage patterns or sensitive organizational queries. Mitigation: add
uniform response delay to mask cache hit/miss timing differences.

**Response Replay:** Cached responses from one context may be inappropriately
served in a different context (e.g., a cached response to a time-sensitive query
may be stale). Mitigation: include temporal context in cache keys for
time-sensitive applications, or disable caching for such query types.

## 4. Mitigation Strategies

1. **PII Detection:** Implement a pre-cache filter using regex or a dedicated
   PII detection model to identify and exclude sensitive prompts from caching.
2. **Audit Logging:** Log all cache operations (writes, hits, invalidations)
   with anonymized request hashes for security auditing without storing content.
3. **User Opt-Out:** Provide a request-level flag (use_cache: false) allowing
   users or applications to bypass caching for sensitive queries.
4. **Encryption at Rest:** Encrypt Redis storage to protect cached content
   from unauthorized access to the cache store.
5. **Access Controls:** Restrict access to the DELETE /cache and GET /metrics
   endpoints to authorized administrators only.

## 5. Compliance Implications

**GDPR (EU):** Article 17 (right to erasure) requires that user data be
deletable on request. Our cache invalidation endpoint supports this. Article 25
(data protection by design) is addressed through our hash-only key design and
TTL-based automatic expiration. Organizations deploying in the EU must ensure
Redis is hosted within EU data residency boundaries.

**Data Residency:** Cloud-hosted Redis instances must comply with applicable
data sovereignty laws. For US federal deployments, FedRAMP-authorized Redis
instances are required. For healthcare applications, PHI must never be cached
without HIPAA-compliant storage and BAA agreements with the cache provider.

**CCPA (California):** Similar to GDPR, users have the right to deletion.
The cache invalidation endpoint satisfies this requirement provided user
requests can be mapped to specific cache keys, which our hashed-key design
makes difficult. A per-user cache namespace with user-controlled invalidation
is recommended for consumer-facing deployments.
