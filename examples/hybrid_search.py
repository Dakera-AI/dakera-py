#!/usr/bin/env python3
"""
Hybrid search example for Dakera Python SDK.

This example demonstrates:
- Indexing documents for full-text search
- Upserting vectors with the same IDs
- Performing hybrid search (vector + text)
- Adjusting vector_weight parameter for search balance
"""

import os

from dakera import DakeraClient, Document


def main():
    client = DakeraClient(
        os.environ.get("DAKERA_API_URL", "http://localhost:3300"),
        api_key=os.environ.get("DAKERA_API_KEY", "dk-mykey"),
    )

    namespace = "hybrid-example"

    # Create namespace with the right dimensions for our sample vectors
    client.create_namespace(namespace, dimensions=5)

    # Sample documents with embeddings (in real usage, generate embeddings with a model)
    documents = [
        {
            "id": "doc1",
            "content": "Machine learning is transforming how we build software applications",
            "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],  # Simulated embedding
            "metadata": {"category": "technology", "year": 2024},
        },
        {
            "id": "doc2",
            "content": "Vector databases enable efficient similarity search at scale",
            "embedding": [0.2, 0.3, 0.4, 0.5, 0.6],
            "metadata": {"category": "technology", "year": 2024},
        },
        {
            "id": "doc3",
            "content": "Natural language processing powers modern search engines",
            "embedding": [0.15, 0.25, 0.35, 0.45, 0.55],
            "metadata": {"category": "technology", "year": 2023},
        },
        {
            "id": "doc4",
            "content": "Deep learning models require significant computational resources",
            "embedding": [0.3, 0.4, 0.5, 0.6, 0.7],
            "metadata": {"category": "technology", "year": 2023},
        },
    ]

    # Index documents for full-text search (may not be supported on all server versions)
    print("Indexing documents for full-text search...")
    fulltext_available = True
    try:
        client.index_documents(
            namespace,
            documents=[
                Document(id=doc["id"], content=doc["content"], metadata=doc["metadata"])
                for doc in documents
            ],
        )
    except Exception as e:
        print(f"  Full-text indexing not supported on this server version: {e}")
        fulltext_available = False

    # Upsert vectors (embeddings)
    print("Upserting vector embeddings...")
    client.upsert(
        namespace,
        vectors=[
            {"id": doc["id"], "values": doc["embedding"], "metadata": doc["metadata"]}
            for doc in documents
        ],
    )

    # Query embedding (simulated - in real usage, embed the query text)
    query_embedding = [0.18, 0.28, 0.38, 0.48, 0.58]
    query_text = "machine learning applications"

    # Pure vector search
    print("\n--- Pure Vector Search ---")
    vector_results = client.query(namespace, vector=query_embedding, top_k=3)
    for r in vector_results.results:
        print(f"  {r.id}: score={r.score:.4f}")

    if fulltext_available:
        # Pure full-text search
        print("\n--- Pure Full-Text Search ---")
        try:
            text_results = client.fulltext_search(namespace, query=query_text, top_k=3)
            for r in text_results:
                print(f"  {r.id}: score={r.score:.4f}")
        except Exception as e:
            print(f"  Full-text search failed: {e}")

        # Hybrid search with different vector_weight values
        print("\n--- Hybrid Search (vector_weight=0.3 - more text) ---")
        try:
            hybrid_results = client.hybrid_search(
                namespace,
                vector=query_embedding,
                query=query_text,
                top_k=3,
                vector_weight=0.3,
            )
            for r in hybrid_results:
                print(
                    f"  {r.id}: combined={r.score:.4f}, "
                    f"vector={r.vector_score:.4f if r.vector_score else 'N/A'}, "
                    f"text={r.text_score:.4f if r.text_score else 'N/A'}"
                )
        except Exception as e:
            print(f"  Hybrid search failed: {e}")

        print("\n--- Hybrid Search (vector_weight=0.5 - balanced) ---")
        try:
            hybrid_results = client.hybrid_search(
                namespace,
                vector=query_embedding,
                query=query_text,
                top_k=3,
                vector_weight=0.5,
            )
            for r in hybrid_results:
                print(
                    f"  {r.id}: combined={r.score:.4f}, "
                    f"vector={r.vector_score:.4f if r.vector_score else 'N/A'}, "
                    f"text={r.text_score:.4f if r.text_score else 'N/A'}"
                )
        except Exception as e:
            print(f"  Hybrid search failed: {e}")

        print("\n--- Hybrid Search (vector_weight=0.7 - more vector) ---")
        try:
            hybrid_results = client.hybrid_search(
                namespace,
                vector=query_embedding,
                query=query_text,
                top_k=3,
                vector_weight=0.7,
            )
            for r in hybrid_results:
                print(
                    f"  {r.id}: combined={r.score:.4f}, "
                    f"vector={r.vector_score:.4f if r.vector_score else 'N/A'}, "
                    f"text={r.text_score:.4f if r.text_score else 'N/A'}"
                )
        except Exception as e:
            print(f"  Hybrid search failed: {e}")

        # Hybrid search with metadata filter
        print("\n--- Hybrid Search with Filter (year=2024) ---")
        try:
            filtered_results = client.hybrid_search(
                namespace,
                vector=query_embedding,
                query=query_text,
                top_k=3,
                vector_weight=0.5,
                filter={"year": {"$eq": 2024}},
            )
            for r in filtered_results:
                print(f"  {r.id}: score={r.score:.4f}")
        except Exception as e:
            print(f"  Filtered hybrid search failed: {e}")
    else:
        print("\nSkipping full-text and hybrid search (not supported on this server)")

    # Cleanup
    print("\nCleaning up...")
    client.delete_namespace(namespace)
    print("Done!")

    client.close()


if __name__ == "__main__":
    main()
