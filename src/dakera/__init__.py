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

try:
    from dakera.async_client import AsyncDakeraClient
except ImportError:
    AsyncDakeraClient = None  # type: ignore[assignment,misc]
from dakera.exceptions import (
    AuthenticationError,
    AuthorizationError,
    ConnectionError,
    DakeraError,
    ErrorCode,
    NotFoundError,
    RateLimitError,
    ServerError,
    ValidationError,
)
from dakera.models import (
    AccessPatternHint,
    AgentNetworkEdge,
    AgentNetworkInfo,
    AgentNetworkNode,
    AgentNetworkStats,
    AgentStats,
    AgentSummary,
    AnalyticsOverview,
    ConfigureNamespaceRequest,
    ConfigureNamespaceResponse,
    ConsolidateResponse,
    CrossAgentNetworkResponse,
    DakeraEvent,
    DeduplicateResponse,
    DistanceMetric,
    Document,
    EmbeddingModel,
    FullTextSearchResult,
    HybridSearchResult,
    IndexStats,
    KnowledgeEdge,
    KnowledgeGraphResponse,
    KnowledgeNode,
    Memory,
    MemoryEvent,
    NamespaceInfo,
    OpStatus,
    QueryResult,
    ReadConsistency,
    RecalledMemory,
    SearchResult,
    Session,
    StalenessConfig,
    StoreMemoryRequest,
    SummarizeResponse,
    Vector,
    VectorMutationOp,
    WarmCacheRequest,
    WarmCacheResponse,
    WarmingPriority,
    WarmingTargetTier,
)

__version__ = "0.6.1"
__all__ = [
    # Clients
    "DakeraClient",
    "AsyncDakeraClient",
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
    # Inference / embedding types (v0.6.0)
    "EmbeddingModel",
    "ConfigureNamespaceRequest",
    "ConfigureNamespaceResponse",
    # SSE Streaming types
    "DakeraEvent",
    "MemoryEvent",
    "OpStatus",
    "VectorMutationOp",
    # Cross-agent network types (DASH-A)
    "AgentNetworkInfo",
    "AgentNetworkNode",
    "AgentNetworkEdge",
    "AgentNetworkStats",
    "CrossAgentNetworkResponse",
    # Exceptions
    "DakeraError",
    "ConnectionError",
    "NotFoundError",
    "ValidationError",
    "RateLimitError",
    "ServerError",
    "AuthenticationError",
    "AuthorizationError",
    "ErrorCode",
]
