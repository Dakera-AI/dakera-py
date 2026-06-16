#!/usr/bin/env python3
"""
Dakera Python SDK — Playground Quickstart

Demonstrates the 4 core memory operations against the Dakera Playground.
Run:
    pip install "dakera>=0.12.2"
    python examples/playground/quickstart.py
"""

import os
import sys

from dakera import DakeraClient

PLAYGROUND_URL = os.environ.get("DAKERA_API_URL", "http://5.75.177.31")
PLAYGROUND_KEY = os.environ.get("DAKERA_API_KEY", "playground-demo")

AGENT_ID = "playground-agent"


def main() -> None:
    client = DakeraClient(PLAYGROUND_URL, api_key=PLAYGROUND_KEY)

    health = client.health()
    print(f"Playground: {health}")

    # -------------------------------------------------------------------------
    # 1. Store memories
    # -------------------------------------------------------------------------
    print("\n--- 1. Store Memories ---")

    mem1 = client.store_memory(
        AGENT_ID,
        content="Dakera provides persistent, decay-weighted memory for AI agents.",
        memory_type="semantic",
        importance=0.9,
        tags=["dakera", "memory", "overview"],
    )
    mem1_id = mem1.get("id") or mem1.get("memory_id")
    print(f"Stored: {mem1_id}")

    mem2 = client.store_memory(
        AGENT_ID,
        content="The recall API returns semantically similar memories ranked by relevance.",
        memory_type="semantic",
        importance=0.8,
        tags=["dakera", "recall", "api"],
    )
    mem2_id = mem2.get("id") or mem2.get("memory_id")
    print(f"Stored: {mem2_id}")

    mem3 = client.store_memory(
        AGENT_ID,
        content="Session scoping lets agents isolate memories per task or conversation.",
        memory_type="episodic",
        importance=0.7,
        tags=["sessions", "isolation"],
    )
    mem3_id = mem3.get("id") or mem3.get("memory_id")
    print(f"Stored: {mem3_id}")

    # -------------------------------------------------------------------------
    # 2. Recall by query (semantic search)
    # -------------------------------------------------------------------------
    print("\n--- 2. Recall by Query ---")

    recalled = client.recall(AGENT_ID, "How does Dakera memory work?", top_k=5)
    print(f"Recalled {len(recalled.memories)} memories:")
    for m in recalled.memories:
        score = f"{m.score:.3f}" if m.score is not None else "n/a"
        print(f"  [{score}] {m.content[:80]}")

    # -------------------------------------------------------------------------
    # 3. Search with filters
    # -------------------------------------------------------------------------
    print("\n--- 3. Search with Filters ---")

    filtered = client.search_memories(
        AGENT_ID,
        query="memory API",
        memory_type="semantic",
        top_k=3,
    )
    print(f"Filtered search ({len(filtered)} results):")
    for m in filtered:
        print(f"  [{m.get('score', 0.0):.3f}] {m.get('content', '')[:80]}")

    # -------------------------------------------------------------------------
    # 4. Knowledge graph link
    # -------------------------------------------------------------------------
    print("\n--- 4. Knowledge Graph Link ---")

    if mem1_id and mem2_id:
        link = client.memory_link(mem1_id, mem2_id, edge_type="related_to")
        print(f"Linked {mem1_id} → {mem2_id}: edge_type={link.edge.edge_type}")
    else:
        print("Skipped KG link (memory IDs unavailable)")

    print("\nPlayground quickstart complete! Visit https://dakera.ai to learn more.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
