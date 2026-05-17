#!/usr/bin/env python3
"""
Vector operations example for Dakera Python SDK.

This example demonstrates:
- Bulk upsert of vectors
- Bulk update and bulk delete
- Vector count and aggregation
- Exporting vectors from a namespace
"""

import os
import sys

from dakera import DakeraClient, Vector


def main():
    # Connect to Dakera server
    client = DakeraClient(
        os.environ.get("DAKERA_API_URL", "http://localhost:3300"),
        api_key=os.environ.get("DAKERA_API_KEY", "dk-mykey"),
    )

    namespace = "vector-ops-example"

    # Create namespace for this example
    try:
        client.create_namespace(namespace, dimensions=128, index_type="hnsw")
        print(f"Created namespace: {namespace}")
    except Exception as e:
        print(f"Namespace may already exist: {e}")

    # --- Bulk Upsert ---
    print("\n=== Bulk Upsert ===")

    # Generate a batch of vectors
    vectors = []
    for i in range(100):
        values = [(i * 0.01 + j * 0.001) for j in range(128)]
        vectors.append(
            Vector(
                id=f"doc-{i:04d}",
                values=values,
                metadata={"batch": "initial", "index": i, "category": f"cat-{i % 5}"},
            )
        )

    client.upsert(namespace, vectors=vectors)
    print(f"Upserted {len(vectors)} vectors")

    # --- Vector Count ---
    print("\n=== Vector Count ===")

    count = client.count_vectors(namespace)
    print(f"Total vectors in namespace: {count}")
    assert count is not None, "expected non-None count result"

    # Count with filter
    filtered_count = client.count_vectors(namespace, filter={"category": "cat-0"})
    print(f"Vectors with category=cat-0: {filtered_count}")

    # --- Bulk Update ---
    print("\n=== Bulk Update ===")

    # Update metadata for a subset of vectors
    updates = [
        {"id": f"doc-{i:04d}", "metadata": {"batch": "updated", "priority": "high"}}
        for i in range(10)
    ]
    update_result = client.bulk_update_vectors(namespace, updates=updates)
    print(f"Bulk update result: {update_result}")
    assert update_result is not None, "expected non-None update result"

    # Verify update by fetching a vector
    fetched = client.fetch(namespace, ids=["doc-0000", "doc-0001"])
    print(f"Fetched after update:")
    for vec in fetched:
        print(f"  - {vec['id']}: metadata={vec.get('metadata')}")

    # --- Aggregate ---
    print("\n=== Aggregation ===")

    # Aggregate vectors by category
    agg_result = client.aggregate(
        namespace,
        group_by="category",
        metrics=["count", "avg_score"],
        vector=[0.05] * 128,
    )
    print(f"Aggregation result: {agg_result}")
    assert agg_result is not None, "expected non-None aggregation result"

    # --- Export ---
    print("\n=== Export Vectors ===")

    # Export first page of vectors
    exported = client.export_vectors(namespace, limit=10, offset=0)
    print(f"Exported {len(exported)} vectors (first page)")
    assert exported is not None, "expected non-None export result"

    for vec in exported[:3]:
        vec_id = vec.get("id", "unknown")
        dims = len(vec.get("values", []))
        print(f"  - {vec_id}: {dims} dimensions")

    # --- Bulk Delete ---
    print("\n=== Bulk Delete ===")

    # Delete vectors by IDs
    ids_to_delete = [f"doc-{i:04d}" for i in range(50, 100)]
    delete_result = client.bulk_delete_vectors(namespace, ids=ids_to_delete)
    print(f"Bulk delete result: {delete_result}")
    assert delete_result is not None, "expected non-None delete result"

    # Delete by filter
    filter_delete = client.delete(namespace, filter={"batch": "initial"})
    print(f"Filter delete result: {filter_delete}")

    # Verify final count
    final_count = client.count_vectors(namespace)
    print(f"Final vector count: {final_count}")

    # --- Cleanup ---
    print("\nCleaning up...")
    client.delete_namespace(namespace)
    print(f"Deleted namespace: {namespace}")

    client.close()
    print("\nDone!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FATAL: {e}", file=sys.stderr)
        sys.exit(1)
