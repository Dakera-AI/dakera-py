#!/usr/bin/env python3
"""
Dakera Python SDK — Memory & Session Operations.

Run:
    python examples/memory.py
"""

import os
import sys

from dakera import BatchRecallRequest, DakeraClient


def main() -> None:
    client = DakeraClient(
        os.environ.get("DAKERA_API_URL", "http://localhost:3000"),
        api_key=os.environ.get("DAKERA_API_KEY", "dk-mykey"),
    )

    agent_id = "agent-demo"

    # -------------------------------------------------------------------------
    # Store memories
    # -------------------------------------------------------------------------
    print("--- Storing Memories ---")

    mem1 = client.store_memory(
        agent_id,
        content="The user prefers concise responses with code examples.",
        memory_type="semantic",
        importance=0.9,
        metadata={"source": "user-feedback"},
    )
    mem1_id = mem1.get("id")
    print(f"Stored memory: {mem1_id}")

    mem2 = client.store_memory(
        agent_id,
        content="User is building a Python microservice with FastAPI.",
        memory_type="episodic",
        importance=0.7,
    )
    mem2_id = mem2.get("id")
    print(f"Stored memory: {mem2_id}")

    # -------------------------------------------------------------------------
    # Recall memories (semantic search)
    # -------------------------------------------------------------------------
    print("\n--- Recalling Memories ---")

    recalled = client.recall(agent_id, "What does the user prefer?", top_k=5)
    for m in recalled.memories:
        score = f"{m.score:.2f}" if m.score is not None else "n/a"
        print(f"  [{score}] {m.memory_type} — {m.content}")

    # -------------------------------------------------------------------------
    # Search memories by type
    # -------------------------------------------------------------------------
    print("\n--- Search Memories (type=semantic) ---")

    searched = client.search_memories(
        agent_id,
        query="user preferences",
        memory_type="semantic",
        top_k=3,
    )
    for m in searched:
        score = f"{m.get('score', 0.0):.2f}"
        print(f"  [{score}] {m.get('content', '')}")

    # -------------------------------------------------------------------------
    # Batch recall (filter-based, no embedding)
    # -------------------------------------------------------------------------
    print("\n--- Batch Recall ---")

    batch_resp = client.batch_recall(
        BatchRecallRequest(agent_id=agent_id, min_importance=0.8)
    )
    print(f"Batch recall found {batch_resp.filtered} memories")

    # -------------------------------------------------------------------------
    # Session management
    # -------------------------------------------------------------------------
    print("\n--- Session Management ---")

    session = client.start_session(agent_id, metadata={"task": "code-review"})
    session_id = session.get("id")
    print(f"Started session: {session_id}")

    # Store a session-scoped memory
    client.store_memory(
        agent_id,
        content="Reviewing PR #42: refactor authentication middleware.",
        session_id=session_id,
    )
    print("Stored session-scoped memory")

    # End the session
    try:
        end_resp = client.end_session(session_id)
        mem_count = end_resp.get("memory_count", "n/a")
        print(f"Ended session (memories: {mem_count})")
    except Exception as e:
        print(f"endSession not fully supported on this server version: {e}")

    # -------------------------------------------------------------------------
    # Agent stats
    # -------------------------------------------------------------------------
    print("\n--- Agent Stats ---")

    try:
        stats = client.agent_stats(agent_id)
        print(f"Agent: {stats.get('agent_id', agent_id)}")
        print(f"  Total memories: {stats.get('total_memories', 'n/a')}")
        print(f"  Total sessions: {stats.get('total_sessions', 'n/a')}")
    except Exception as e:
        print(f"Agent stats not fully supported: {e}")

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------
    for mid in [mem1_id, mem2_id]:
        if mid:
            try:
                client.forget(agent_id, mid)
            except Exception:
                pass
    print("\nCleaned up memories")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FATAL: {e}", file=sys.stderr)
        sys.exit(1)
