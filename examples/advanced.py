#!/usr/bin/env python3
"""
Dakera Python SDK — Advanced Features.

Covers: text auto-embedding, full-text search, hybrid search,
knowledge graph, feedback loop, analytics.

Run:
    python examples/advanced.py
"""

import os
import sys

from dakera import DakeraClient


def main() -> None:
    client = DakeraClient(
        os.environ.get("DAKERA_API_URL", "http://localhost:3000"),
        api_key=os.environ.get("DAKERA_API_KEY", "dk-mykey"),
    )

    namespace = "example-advanced"

    # Create namespace (3 dims for vector ops; text embedding auto-sizes)
    try:
        client.create_namespace(namespace, dimensions=3)
    except Exception:
        pass  # already exists

    # -------------------------------------------------------------------------
    # Text auto-embedding (server generates vectors)
    # -------------------------------------------------------------------------
    print("--- Text Auto-Embedding ---")

    text_resp = client.upsert_text(
        namespace,
        documents=[
            {"id": "doc1", "text": "Rust memory safety prevents data races at compile time."},
            {"id": "doc2", "text": "Go goroutines enable lightweight concurrency patterns."},
            {"id": "doc3", "text": "Python asyncio provides cooperative multitasking."},
        ],
    )
    print(f"Upserted {text_resp.upserted_count} text documents")

    text_results = client.query_text(namespace, "concurrent programming", top_k=3)
    print("Text search results:")
    for r in text_results.results:
        print(f"  {r.id}: {r.text} (score: {r.score:.4f})")

    # -------------------------------------------------------------------------
    # Full-text search (BM25)
    # -------------------------------------------------------------------------
    print("\n--- Full-Text Search ---")

    client.index_documents(
        namespace,
        documents=[
            {"id": "ft1", "text": "Vector databases enable semantic search over embeddings."},
            {"id": "ft2", "text": "BM25 ranking uses term frequency and document length."},
            {"id": "ft3", "text": "Hybrid search combines vector similarity with keyword matching."},
        ],
    )

    ft_results = client.fulltext_search(namespace, "vector search", top_k=5)
    print("Full-text results:")
    for r in ft_results:
        print(f"  {r.id}: score {r.score:.4f}")

    # -------------------------------------------------------------------------
    # Hybrid search (vector + BM25)
    # -------------------------------------------------------------------------
    print("\n--- Hybrid Search ---")

    hybrid_results = client.hybrid_search(
        namespace,
        query="semantic search",
        top_k=5,
        vector_weight=0.7,
    )
    print("Hybrid results:")
    for r in hybrid_results:
        print(f"  {r.id}: score {r.score:.4f}")

    # -------------------------------------------------------------------------
    # Knowledge graph
    # -------------------------------------------------------------------------
    print("\n--- Knowledge Graph ---")

    agent_id = "agent-demo"
    m1_id: str | None = None
    m2_id: str | None = None

    try:
        m1 = client.store_memory(
            agent_id,
            content="User is a senior backend engineer.",
            memory_type="semantic",
            importance=0.9,
        )
        m1_id = m1.get("id")

        m2 = client.store_memory(
            agent_id,
            content="User works primarily with Go and Rust.",
            memory_type="semantic",
            importance=0.8,
        )
        m2_id = m2.get("id")

        client.memory_link(m1_id, m2_id, "related_to")

        graph = client.memory_graph(m1_id, depth=2)
        print(f"Graph nodes: {len(graph.nodes)}, edges: {len(graph.edges)}")

        path = client.memory_path(m1_id, m2_id)
        print(f"Path length: {len(path.path)} nodes ({path.hops} hops)")
    except Exception as e:
        print(f"Knowledge graph not fully supported on this server version: {e}")

    # -------------------------------------------------------------------------
    # Feedback loop
    # -------------------------------------------------------------------------
    print("\n--- Feedback Loop ---")

    try:
        if m1_id:
            client.feedback_memory(m1_id, agent_id, "upvote")
            history = client.get_memory_feedback_history(m1_id)
            print(f"Feedback entries: {len(history.entries)}")

        summary = client.get_agent_feedback_summary(agent_id)
        print(f"Agent feedback — upvotes: {summary.upvotes}, downvotes: {summary.downvotes}")
    except Exception as e:
        print(f"Feedback not fully supported on this server version: {e}")

    # -------------------------------------------------------------------------
    # Analytics
    # -------------------------------------------------------------------------
    print("\n--- Analytics ---")

    try:
        overview = client.analytics_overview()
        print(f"Total queries: {overview.get('total_queries', 'n/a')}")
    except Exception as e:
        print(f"Analytics not fully supported: {e}")

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------
    try:
        for mid in [m1_id, m2_id]:
            if mid:
                client.forget(agent_id, mid)
    except Exception:
        pass
    client.delete_namespace(namespace)
    print("\nCleaned up")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FATAL: {e}", file=sys.stderr)
        sys.exit(1)
