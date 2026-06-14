#!/usr/bin/env python3
"""
Full-Text Search — index documents, search, stats, delete.

Run:
    python examples/fulltext_search.py
"""

import os
import sys

from dakera import DakeraClient


def main() -> None:
    client = DakeraClient(
        os.environ.get("DAKERA_API_URL", "http://localhost:3000"),
        api_key=os.environ.get("DAKERA_API_KEY", "dk-mykey"),
    )

    health = client.health()
    print(f"Server: {health.get('version')} (status: {health.get('status')})")

    namespace = "example-fulltext"

    # Create namespace for full-text indexing
    try:
        client.create_namespace(namespace, dimensions=3)
    except Exception:
        pass  # already exists

    # --- Index Documents ---
    print("\n--- Indexing Documents ---")
    index_resp = client.index_documents(
        namespace,
        documents=[
            {"id": "doc1", "text": "The quick brown fox jumps over the lazy dog", "metadata": {"topic": "animals"}},
            {"id": "doc2", "text": "Machine learning enables computers to learn from data", "metadata": {"topic": "tech"}},
            {"id": "doc3", "text": "Neural networks are inspired by biological neurons", "metadata": {"topic": "tech"}},
            {"id": "doc4", "text": "The fox ran swiftly through the dense forest", "metadata": {"topic": "animals"}},
            {"id": "doc5", "text": "Deep learning is a subset of machine learning algorithms", "metadata": {"topic": "tech"}},
        ],
    )
    if not index_resp or index_resp.get("indexed_count") != 5:
        raise AssertionError(f"expected 5 indexed, got {index_resp}")
    print(f"Indexed {index_resp['indexed_count']} documents")

    # --- Full-Text Search ---
    print('\n--- Search: "machine learning" ---')
    results = client.fulltext_search(namespace, "machine learning", top_k=5)
    if not results:
        raise AssertionError("expected non-empty search results")
    for r in results:
        print(f"  ID: {r.id}, Score: {r.score:.4f}")

    # Search with metadata filter
    print('\n--- Filtered Search: "fox" (topic=animals) ---')
    filtered = client.fulltext_search(
        namespace,
        "fox",
        top_k=5,
        filter={"topic": {"$eq": "animals"}},
    )
    for r in filtered:
        print(f"  ID: {r.id}, Score: {r.score:.4f}")
    if len(filtered) < 1:
        raise AssertionError('expected at least 1 result for "fox" in animals')

    # --- Index Stats ---
    print("\n--- Full-Text Index Stats ---")
    stats = client.fulltext_stats(namespace)
    print(f"Documents indexed: {stats.document_count}")
    print(f"Unique terms: {stats.unique_terms}")
    print(f"Avg doc length: {stats.avg_doc_length:.2f}")

    # --- Delete Documents from Index ---
    print("\n--- Delete Documents ---")
    del_resp = client.fulltext_delete(namespace, ["doc1", "doc2"])
    print(f"Deleted {del_resp.get('deleted_count', 0)} documents from full-text index")

    # Verify reduced count
    stats_after = client.fulltext_stats(namespace)
    print(f"Documents remaining: {stats_after.document_count}")

    # Cleanup
    client.delete_namespace(namespace)
    print("\nNamespace deleted. Full-text search example completed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FATAL: {e}", file=sys.stderr)
        sys.exit(1)
