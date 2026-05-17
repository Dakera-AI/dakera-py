#!/usr/bin/env python3
"""
Knowledge graph example for Dakera Python SDK.

This example demonstrates:
- Building a knowledge graph from agent memories
- Traversing graph relationships
- Querying the knowledge graph
- Finding paths between entities
- Exporting the knowledge graph
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

    agent_id = "kg-example-agent"

    # Store some memories to build graph from
    print("=== Storing Memories for Knowledge Graph ===")

    memories = [
        {
            "text": "Alice is the CTO of Acme Corp. She leads the engineering team.",
            "metadata": {"source": "meeting-notes", "date": "2026-01-15"},
        },
        {
            "text": "Bob reports to Alice and works on the backend infrastructure.",
            "metadata": {"source": "org-chart", "date": "2026-01-10"},
        },
        {
            "text": "Acme Corp is building a real-time analytics platform using Rust.",
            "metadata": {"source": "product-brief", "date": "2026-02-01"},
        },
        {
            "text": "The analytics platform depends on the backend infrastructure Bob maintains.",
            "metadata": {"source": "architecture-doc", "date": "2026-02-05"},
        },
        {
            "text": "Carol joined the ML team at Acme Corp to build recommendation models.",
            "metadata": {"source": "hiring-update", "date": "2026-03-01"},
        },
    ]

    memory_ids = []
    for mem in memories:
        result = client.store_memory(
            agent_id=agent_id,
            text=mem["text"],
            metadata=mem["metadata"],
        )
        mid = result.get("memory_id") or result.get("id")
        memory_ids.append(mid)
        print(f"  Stored memory: {mid}")

    assert len(memory_ids) == 5, "expected 5 memories stored"

    # --- Build Knowledge Graph ---
    print("\n=== Building Knowledge Graph ===")

    kg_result = client.knowledge_graph(agent_id=agent_id)
    print(f"Knowledge graph built: {kg_result}")
    assert kg_result is not None, "expected non-None knowledge graph result"

    # Get full knowledge graph structure
    full_kg = client.full_knowledge_graph(agent_id=agent_id)
    print(f"Full KG nodes: {len(full_kg.get('nodes', []))}")
    print(f"Full KG edges: {len(full_kg.get('edges', []))}")

    for node in full_kg.get("nodes", [])[:5]:
        print(f"  Node: {node.get('label', node.get('id'))}")

    for edge in full_kg.get("edges", [])[:5]:
        print(f"  Edge: {edge.get('source')} --[{edge.get('relation')}]--> {edge.get('target')}")

    # --- Traverse Graph ---
    print("\n=== Graph Traversal ===")

    # Traverse from a specific entity
    traversal = client.memory_graph(
        agent_id=agent_id,
        entity="Alice",
        depth=2,
    )
    print(f"Traversal from 'Alice' (depth=2): {traversal}")
    assert traversal is not None, "expected non-None traversal result"

    # --- Query Knowledge Graph ---
    print("\n=== Knowledge Graph Query ===")

    # Query for relationships
    query_result = client.knowledge_query(
        agent_id=agent_id,
        query="Who works at Acme Corp?",
        top_k=5,
    )
    print(f"KG query result: {query_result}")
    assert query_result is not None, "expected non-None KG query result"

    # --- Find Path ---
    print("\n=== Path Finding ===")

    # Find path between two entities
    path_result = client.knowledge_path(
        agent_id=agent_id,
        source="Alice",
        target="analytics platform",
    )
    print(f"Path from Alice to analytics platform: {path_result}")
    assert path_result is not None, "expected non-None path result"

    # Also try memory_path for direct memory links
    if len(memory_ids) >= 2 and memory_ids[0] and memory_ids[1]:
        mem_path = client.memory_path(memory_ids[0], memory_ids[1])
        print(f"Memory path: {mem_path}")

    # --- Export Knowledge Graph ---
    print("\n=== Export Knowledge Graph ===")

    export = client.knowledge_export(agent_id=agent_id, format="json")
    print(f"Exported KG: {type(export)}")
    assert export is not None, "expected non-None export result"

    if isinstance(export, dict):
        print(f"  Nodes: {len(export.get('nodes', []))}")
        print(f"  Edges: {len(export.get('edges', []))}")

    # --- Cleanup ---
    print("\nCleaning up memories...")
    for mid in memory_ids:
        if mid:
            client.forget(agent_id, mid)
    print(f"Deleted {len(memory_ids)} memories")

    client.close()
    print("\nDone!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FATAL: {e}", file=sys.stderr)
        sys.exit(1)
