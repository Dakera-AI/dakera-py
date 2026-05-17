#!/usr/bin/env python3
"""
Ops diagnostics example for Dakera Python SDK.

This example demonstrates:
- Running server diagnostics
- Listing and inspecting background jobs
- Triggering compaction
- Viewing and clearing cache stats
"""

import os
import sys

from dakera import DakeraClient


def main():
    # Connect to Dakera server
    client = DakeraClient(
        os.environ.get("DAKERA_API_URL", "http://localhost:3300"),
        api_key=os.environ.get("DAKERA_API_KEY", "dk-mykey"),
    )

    # Check server health (readiness + liveness)
    health = client.health()
    print(f"Health: {health}")

    ready = client.health_ready()
    print(f"Readiness: {ready}")

    live = client.health_live()
    print(f"Liveness: {live}")

    # --- Diagnostics ---
    print("\n=== Diagnostics ===")

    diagnostics = client.ops_diagnostics()
    print(f"Diagnostics: {diagnostics}")
    assert diagnostics is not None, "expected non-None diagnostics"

    # Server stats and Prometheus metrics
    stats = client.ops_stats()
    print(f"Ops stats: {stats}")
    assert stats is not None, "expected non-None ops stats"

    metrics = client.ops_metrics()
    print(f"Metrics (first 200 chars): {metrics[:200]}...")

    # --- Background Jobs ---
    print("\n=== Background Jobs ===")

    jobs = client.ops_list_jobs()
    print(f"Active jobs: {len(jobs)}")
    for job in jobs[:5]:
        job_id = job.get("id") or job.get("job_id")
        status = job.get("status", "unknown")
        print(f"  - Job {job_id}: {status}")

        # Inspect individual job details
        if job_id:
            detail = client.ops_get_job(job_id)
            print(f"    Detail: {detail}")

    # --- Compaction ---
    print("\n=== Compaction ===")

    # List namespaces to pick one for compaction
    namespaces = client.list_namespaces()
    if namespaces:
        ns_name = namespaces[0].name
        print(f"Triggering compaction on namespace: {ns_name}")

        compact_result = client.ops_compact(namespace=ns_name)
        print(f"Compaction result: {compact_result}")
        assert compact_result is not None, "expected non-None compaction result"

        # Also check index stats after compaction
        idx_stats = client.get_index_stats(ns_name)
        print(f"Index stats for {ns_name}: segments={idx_stats.segments}")
    else:
        print("No namespaces available for compaction demo")

    # --- Cache ---
    print("\n=== Cache Stats ===")

    cache = client.cache_stats()
    print(f"Cache stats: {cache}")
    assert cache is not None, "expected non-None cache stats"

    # Clear cache for a specific namespace (or all)
    if namespaces:
        clear_result = client.cache_clear(namespace=namespaces[0].name)
        print(f"Cache cleared for {namespaces[0].name}: {clear_result}")
    else:
        clear_result = client.cache_clear()
        print(f"Global cache cleared: {clear_result}")

    # --- Background Activity ---
    print("\n=== Background Activity ===")

    activity = client.background_activity()
    print(f"Background activity: {activity}")

    # --- Storage Tiers ---
    print("\n=== Storage Tier Overview ===")

    storage = client.storage_tier_overview()
    print(f"Storage tiers: {storage}")

    client.close()
    print("\nDone!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FATAL: {e}", file=sys.stderr)
        sys.exit(1)
