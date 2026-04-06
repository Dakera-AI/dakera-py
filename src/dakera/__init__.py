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
    AgentFeedbackSummary,
    AgentNetworkEdge,
    AgentNetworkInfo,
    AgentNetworkNode,
    AgentNetworkStats,
    AgentStats,
    AgentSummary,
    AnalyticsOverview,
    # OBS-1
    AuditEvent,
    AuditExportResponse,
    AuditListResponse,
    BatchForgetRequest,
    BatchForgetResponse,
    BatchMemoryFilter,
    BatchRecallRequest,
    BatchRecallResponse,
    ConfigureNamespaceRequest,
    ConfigureNamespaceResponse,
    ConsolidateResponse,
    # CE-6
    ConsolidationConfig,
    ConsolidationLogEntry,
    CreateNamespaceKeyResponse,
    CrossAgentNetworkResponse,
    DakeraEvent,
    DeduplicateResponse,
    DistanceMetric,
    Document,
    EdgeType,
    EmbeddingModel,
    EntityExtractionResponse,
    ExtractedEntity,
    # ODE-2
    ExtractEntitiesResponse,
    ExtractionProviderInfo,
    # EXT-1
    ExtractionResult,
    FeedbackHealthResponse,
    FeedbackHistoryEntry,
    FeedbackHistoryResponse,
    FeedbackResponse,
    FeedbackSignal,
    FullTextSearchResult,
    GraphEdge,
    GraphExport,
    GraphLinkResponse,
    GraphNode,
    GraphPath,
    HybridSearchResult,
    IndexStats,
    # KG-2
    KgExportResponse,
    KgPathResponse,
    KgQueryResponse,
    KnowledgeEdge,
    KnowledgeGraphResponse,
    KnowledgeNode,
    ListNamespaceKeysResponse,
    Memory,
    MemoryEntitiesResponse,
    MemoryEvent,
    MemoryExportResponse,
    MemoryGraph,
    # DX-1
    MemoryImportResponse,
    # COG-1
    MemoryPolicy,
    NamespaceInfo,
    NamespaceKeyInfo,
    NamespaceKeyUsageResponse,
    NamespaceNerConfig,
    OdeEntity,
    OpStatus,
    QueryResult,
    RateLimitHeaders,
    ReadConsistency,
    RecalledMemory,
    # COG-2
    RecallResponse,
    RetryConfig,
    # OBS-2
    KpiSnapshot,
    # SEC-3
    RotateEncryptionKeyRequest,
    RotateEncryptionKeyResponse,
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

__version__ = "0.9.9"
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
    # Retry & timeout configuration
    "RetryConfig",
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
    "RecallResponse",
    "ConsolidateResponse",
    # Batch memory operations (CE-2)
    "BatchMemoryFilter",
    "BatchRecallRequest",
    "BatchRecallResponse",
    "BatchForgetRequest",
    "BatchForgetResponse",
    # Rate-limit headers (OPS-1)
    "RateLimitHeaders",
    # Session types
    "Session",
    # Agent types
    "AgentSummary",
    "AgentStats",
    # Knowledge Graph types (internal graph — knowledge.rs)
    "KnowledgeNode",
    "KnowledgeEdge",
    "KnowledgeGraphResponse",
    "SummarizeResponse",
    "DeduplicateResponse",
    # Memory Knowledge Graph types (CE-5 / SDK-9)
    "EdgeType",
    "GraphEdge",
    "GraphNode",
    "MemoryGraph",
    "GraphPath",
    "GraphLinkResponse",
    "GraphExport",
    # KG-2: Graph Query & Export types
    "KgQueryResponse",
    "KgPathResponse",
    "KgExportResponse",
    # Entity Extraction types (CE-4 / GLiNER)
    "NamespaceNerConfig",
    "ExtractedEntity",
    "EntityExtractionResponse",
    "MemoryEntitiesResponse",
    # Memory Feedback Loop (INT-1)
    "FeedbackSignal",
    "FeedbackHistoryEntry",
    "FeedbackResponse",
    "FeedbackHistoryResponse",
    "AgentFeedbackSummary",
    "FeedbackHealthResponse",
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
    # Namespace API Keys (SEC-1)
    "NamespaceKeyInfo",
    "CreateNamespaceKeyResponse",
    "ListNamespaceKeysResponse",
    "NamespaceKeyUsageResponse",
    # DBSCAN Adaptive Consolidation (CE-6)
    "ConsolidationConfig",
    "ConsolidationLogEntry",
    # Memory Import / Export (DX-1)
    "MemoryImportResponse",
    "MemoryExportResponse",
    # Business-Event Audit Log (OBS-1)
    "AuditEvent",
    "AuditListResponse",
    "AuditExportResponse",
    # External Extraction Providers (EXT-1)
    "ExtractionResult",
    "ExtractionProviderInfo",
    # AES-256-GCM Encryption Key Rotation (SEC-3)
    "RotateEncryptionKeyRequest",
    "RotateEncryptionKeyResponse",
    # GLiNER Entity Extraction via ODE sidecar (ODE-2)
    "OdeEntity",
    "ExtractEntitiesResponse",
    # COG-1: Cognitive Memory Lifecycle
    "MemoryPolicy",
    # OBS-2: Product KPI Snapshot
    "KpiSnapshot",
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
