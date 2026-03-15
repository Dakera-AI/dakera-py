"""
Dakera Python SDK

A high-performance Python client for Dakera AI memory platform.

Example usage:
    >>> from dakera import DakeraClient
    >>> client = DakeraClient("http://localhost:3000")
    >>> client.upsert("my-namespace", vectors=[
    ...     {"id": "vec1", "values": [0.1, 0.2, 0.3]}
    ... ])
    >>> results = client.query("my-namespace", vector=[0.1, 0.2, 0.3], top_k=10)
"""

from dakera.client import DakeraClient
from dakera.exceptions import (
    ConnectionError,
    DakeraError,
    NotFoundError,
    RateLimitError,
    ServerError,
    ValidationError,
)
from dakera.models import (
    AccessPatternHint,
    AgentStats,
    # Agent types
    AgentSummary,
    # Analytics types
    AnalyticsOverview,
    ConsolidateResponse,
    DeduplicateResponse,
    DistanceMetric,
    Document,
    FullTextSearchResult,
    HybridSearchResult,
    IndexStats,
    KnowledgeEdge,
    KnowledgeGraphResponse,
    # Knowledge Graph types
    KnowledgeNode,
    Memory,
    NamespaceInfo,
    QueryResult,
    # Consistency types
    ReadConsistency,
    RecalledMemory,
    SearchResult,
    # Session types
    Session,
    StalenessConfig,
    # Memory types
    StoreMemoryRequest,
    SummarizeResponse,
    # Core types
    Vector,
    WarmCacheRequest,
    WarmCacheResponse,
    # Cache warming types
    WarmingPriority,
    WarmingTargetTier,
)

__version__ = "0.1.0"
__all__ = [
    # Client
    "DakeraClient",
    # Models
    "Vector",
    "QueryResult",
    "SearchResult",
    "NamespaceInfo",
    "IndexStats",
    "Document",
    "FullTextSearchResult",
    "HybridSearchResult",
    # Consistency types
    "ReadConsistency",
    "DistanceMetric",
    "StalenessConfig",
    # Cache warming types
    "WarmingPriority",
    "WarmingTargetTier",
    "AccessPatternHint",
    "WarmCacheRequest",
    "WarmCacheResponse",
    # Memory types
    "StoreMemoryRequest",
    "Memory",
    "RecalledMemory",
    "ConsolidateResponse",
    # Session types
    "Session",
    # Agent types
    "AgentSummary",
    "AgentStats",
    # Knowledge Graph types
    "KnowledgeNode",
    "KnowledgeEdge",
    "KnowledgeGraphResponse",
    "SummarizeResponse",
    "DeduplicateResponse",
    # Analytics types
    "AnalyticsOverview",
    # Exceptions
    "DakeraError",
    "ConnectionError",
    "NotFoundError",
    "ValidationError",
    "RateLimitError",
    "ServerError",
]
