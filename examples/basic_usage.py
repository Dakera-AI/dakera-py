#!/usr/bin/env python3
"""
Basic usage example for Dakera Python SDK.

This example demonstrates:
- Connecting to Dakera
- Creating a namespace
- Upserting vectors
- Querying vectors
- Deleting vectors
"""

from dakera import DakeraClient, Vector


def main():
    # Connect to Dakera server
    client = DakeraClient("http://localhost:3000")

    # Check server health
    health = client.health()
    print(f"Server status: {health}")

    # Create a namespace (optional - auto-created on first upsert)
    try:
        client.create_namespace(
            "example-namespace",
            dimensions=3,
            index_type="flat",
        )
        print("Created namespace: example-namespace")
    except Exception as e:
        print(f"Namespace may already exist: {e}")

    # Upsert vectors using dictionaries
    print("\nUpserting vectors...")
    client.upsert(
        "example-namespace",
        vectors=[
            {"id": "vec1", "values": [0.1, 0.2, 0.3], "metadata": {"label": "first"}},
            {"id": "vec2", "values": [0.4, 0.5, 0.6], "metadata": {"label": "second"}},
            {"id": "vec3", "values": [0.7, 0.8, 0.9], "metadata": {"label": "third"}},
        ],
    )

    # Upsert using Vector dataclass
    client.upsert(
        "example-namespace",
        vectors=[
            Vector(id="vec4", values=[0.2, 0.3, 0.4], metadata={"label": "fourth"}),
        ],
    )
    print("Upserted 4 vectors")

    # Query similar vectors
    print("\nQuerying vectors...")
    results = client.query(
        "example-namespace",
        vector=[0.1, 0.2, 0.3],
        top_k=3,
        include_metadata=True,
    )

    print(f"Found {len(results.results)} results:")
    for result in results.results:
        print(f"  - {result.id}: score={result.score:.4f}, metadata={result.metadata}")

    # Query with metadata filter
    print("\nQuerying with filter...")
    filtered_results = client.query(
        "example-namespace",
        vector=[0.5, 0.5, 0.5],
        top_k=10,
        filter={"label": {"$in": ["first", "second"]}},
    )

    print(f"Filtered results: {len(filtered_results.results)}")
    for result in filtered_results.results:
        print(f"  - {result.id}: score={result.score:.4f}")

    # Fetch vectors by ID
    print("\nFetching vectors by ID...")
    vectors = client.fetch("example-namespace", ids=["vec1", "vec2"])
    for v in vectors:
        print(f"  - {v.id}: values={v.values[:3]}...")

    # Get namespace stats
    print("\nNamespace stats:")
    info = client.get_namespace("example-namespace")
    print(f"  - Name: {info.name}")
    print(f"  - Vector count: {info.vector_count}")
    print(f"  - Dimensions: {info.dimensions}")

    # Delete specific vectors
    print("\nDeleting vec4...")
    client.delete("example-namespace", ids=["vec4"])

    # Cleanup - delete namespace
    print("\nCleaning up...")
    client.delete_namespace("example-namespace")
    print("Deleted namespace: example-namespace")

    client.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
