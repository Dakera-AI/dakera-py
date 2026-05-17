#!/usr/bin/env python3
"""
Analytics example for Dakera Python SDK.

This example demonstrates:
- Agent statistics and KPIs
- Analytics overview (latency, throughput, storage)
- Session management and listing
- Memory type stats
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

    agent_id = "analytics-example-agent"

    # Store a few memories so we have data to analyze
    print("=== Setting Up Test Data ===")

    # Start a session
    session = client.start_session(agent_id=agent_id, metadata={"purpose": "analytics-demo"})
    session_id = session.get("session_id") or session.get("id")
    print(f"Started session: {session_id}")
    assert session_id is not None, "expected non-None session_id"

    # Store memories within the session
    for i in range(5):
        client.store_memory(
            agent_id=agent_id,
            text=f"Analytics test memory #{i}: monitoring system performance metrics.",
            metadata={"session_id": session_id, "index": i, "topic": "performance"},
        )
    print("Stored 5 test memories")

    # End the session
    client.end_session(session_id, summary="Analytics demo session completed")
    print(f"Ended session: {session_id}")

    # --- Agent Stats ---
    print("\n=== Agent Stats ===")

    stats = client.agent_stats(agent_id)
    print(f"Agent stats: {stats}")
    assert stats is not None, "expected non-None agent stats"

    # List agent memories
    agent_mems = client.agent_memories(agent_id, limit=10)
    print(f"Agent memories count: {len(agent_mems)}")

    # --- KPIs ---
    print("\n=== KPIs ===")

    kpis = client.get_kpis()
    print(f"KPI snapshot: {kpis}")
    assert kpis is not None, "expected non-None KPI snapshot"

    # --- Analytics Overview ---
    print("\n=== Analytics Overview ===")

    overview = client.analytics_overview()
    print(f"Overview: {overview}")
    assert overview is not None, "expected non-None analytics overview"

    # Latency metrics
    latency = client.analytics_latency()
    print(f"Latency: {latency}")

    # Throughput metrics
    throughput = client.analytics_throughput()
    print(f"Throughput: {throughput}")

    # Storage metrics
    storage = client.analytics_storage()
    print(f"Storage: {storage}")

    # --- Sessions ---
    print("\n=== Sessions ===")

    # List sessions for agent
    sessions = client.agent_sessions(agent_id, limit=10)
    print(f"Agent sessions: {len(sessions)}")
    for s in sessions[:3]:
        sid = s.get("session_id") or s.get("id")
        print(f"  - Session: {sid}")

    # Get session details
    if session_id:
        session_detail = client.get_session(session_id)
        print(f"Session detail: {session_detail}")

        # List memories in session
        session_mems = client.session_memories(session_id)
        print(f"Memories in session: {len(session_mems)}")

    # List all sessions
    all_sessions = client.list_sessions(limit=5)
    print(f"Total sessions (up to 5): {len(all_sessions)}")

    # --- Memory Type Stats ---
    print("\n=== Memory Type Stats ===")

    type_stats = client.memory_type_stats()
    print(f"Memory type stats: {type_stats}")

    # --- Feedback Summary ---
    print("\n=== Feedback Summary ===")

    feedback_summary = client.get_agent_feedback_summary(agent_id)
    print(f"Feedback summary: {feedback_summary}")

    # --- Cleanup ---
    print("\nCleaning up...")
    agent_mems = client.agent_memories(agent_id, limit=100)
    for mem in agent_mems:
        mid = mem.get("memory_id") or mem.get("id")
        if mid:
            client.forget(agent_id, mid)
    print(f"Deleted {len(agent_mems)} memories")

    client.close()
    print("\nDone!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FATAL: {e}", file=sys.stderr)
        sys.exit(1)
