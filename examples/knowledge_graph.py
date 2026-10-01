#!/usr/bin/env python3
"""
Knowledge graph example for Dakera Python SDK.

This example demonstrates:
- Linking memories explicitly
- Building a knowledge graph from a seed memory
- Traversing graph relationships
- Querying the knowledge graph
- Finding paths between memories
- Exporting the knowledge graph
- Summarizing memories
"""

import os
import sys

from dakera import DakeraClient


def main():
    # Connect to Dakera server
    client = DakeraClient(
        os.environ.get("DAKERA_API_URL", "http://localhost:3000"),
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
            content=mem["text"],
            metadata=mem["metadata"],
        )
        mid = result["id"]
        memory_ids.append(mid)
        print(f"  Stored memory: {mid}")

    assert len(memory_ids) == 5, "expected 5 memories stored"

    # --- Explicit links (recorded as `linked_by` edges) ---
    print("\n=== Linking Memories ===")
    alice, bob, platform, depends, _carol = memory_ids
    for src, dst, label in (
        (bob, alice, "reports to"),
        (depends, platform, "describes"),
        (depends, bob, "maintained by"),
    ):
        link = client.memory_link(src, dst, agent_id=agent_id, label=label)
        print(f"  {link.from_id} --[{link.edge_type}: {label}]--> {link.to_id}")

    # --- Build a similarity graph from a seed memory ---
    print("\n=== Building Knowledge Graph ===")
    kg_result = client.knowledge_graph(agent_id=agent_id, memory_id=alice, depth=2)
    print(f"Knowledge graph from Alice: {kg_result.get('total_nodes', 0)} nodes")

    full_kg = client.full_knowledge_graph(agent_id=agent_id)
    print(f"Full KG nodes: {len(full_kg.get('nodes', []))}")
    print(f"Full KG edges: {len(full_kg.get('edges', []))}")

    # --- Traverse Graph ---
    print("\n=== Graph Traversal ===")
    traversal = client.memory_graph(depends, depth=2)
    print(f"Traversal from {depends}: {traversal.node_count} nodes, {len(traversal.edges)} edges")
    for edge in traversal.edges[:5]:
        print(f"  {edge.source_id} --[{edge.edge_type.value}]--> {edge.target_id}")
    assert traversal.edges, "expected the explicit links in the traversal"

    # --- Query Knowledge Graph ---
    print("\n=== Knowledge Graph Query ===")
    query_result = client.knowledge_query(agent_id=agent_id, edge_type="linked_by")
    print(f"linked_by edges: {query_result.edge_count}")
    assert query_result.edge_count >= 3, "expected the three explicit links"

    # --- Find Path ---
    print("\n=== Path Finding ===")
    path_result = client.knowledge_path(agent_id=agent_id, from_id=depends, to_id=alice)
    print(f"Path {' -> '.join(path_result.path)} ({path_result.hop_count} hops)")

    mem_path = client.memory_path(depends, platform)
    print(f"Memory path: {' -> '.join(mem_path.path)} ({mem_path.hops} hops)")

    # --- Export Knowledge Graph ---
    print("\n=== Export Knowledge Graph ===")
    export = client.knowledge_export(agent_id=agent_id, format="json")
    print(f"Exported KG: {export.node_count} nodes, {export.edge_count} edges")
    agent_export = client.agent_graph_export(agent_id)
    print(f"Agent graph export ({agent_export.namespace}): {agent_export.edge_count} edges")

    # --- Entities attached to a memory ---
    entities = client.memory_entities(alice)
    print(f"Entities on {entities.memory_id}: {[e.value for e in entities.entities]}")

    # --- Summarize two memories into one ---
    summary = client.summarize(agent_id, memory_ids=[alice, bob])
    print(f"Summary memory: {summary.get('summary_memory', {}).get('id')}")
    if summary.get("summary_memory", {}).get("id"):
        memory_ids.append(summary["summary_memory"]["id"])

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
