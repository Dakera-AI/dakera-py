"""
Dakera SDK Data Models

Dataclasses representing Dakera data structures.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Union

# ============================================================================
# Consistency & Query Types (Turbopuffer-inspired)
# ============================================================================


class ReadConsistency(str, Enum):
    """Read consistency level for queries."""

    STRONG = "strong"
    """Always read from primary/leader node - guarantees latest data."""

    EVENTUAL = "eventual"
    """Read from any replica - may return slightly stale data but faster."""

    BOUNDED_STALENESS = "bounded_staleness"
    """Read from replicas within staleness bounds."""


class DistanceMetric(str, Enum):
    """Distance metric for similarity search."""

    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"


# ============================================================================
# Retry & Timeout Configuration
# ============================================================================


@dataclass
class RetryConfig:
    """Configuration for request retry behavior with exponential backoff."""

    max_retries: int = 3
    """Maximum number of retry attempts (default: 3)."""

    base_delay: float = 0.1
    """Base delay in seconds before first retry (default: 0.1)."""

    max_delay: float = 60.0
    """Maximum delay in seconds between retries (default: 60.0)."""

    jitter: bool = True
    """Whether to add random jitter to backoff delay (default: True)."""


@dataclass
class StalenessConfig:
    """Configuration for bounded staleness reads."""

    max_staleness_ms: int = 5000
    """Maximum acceptable staleness in milliseconds."""

    def to_dict(self) -> dict[str, Any]:
        return {"max_staleness_ms": self.max_staleness_ms}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StalenessConfig":
        return cls(max_staleness_ms=data.get("max_staleness_ms", 5000))


# ============================================================================
# Cache Warming Types (Turbopuffer-inspired)
# ============================================================================


class WarmingPriority(str, Enum):
    """Priority level for cache warming operations."""

    CRITICAL = "critical"
    """Highest priority - warm immediately, preempt other operations."""

    HIGH = "high"
    """High priority - warm soon."""

    NORMAL = "normal"
    """Normal priority (default)."""

    LOW = "low"
    """Low priority - warm when resources available."""

    BACKGROUND = "background"
    """Background priority - warm during idle time only."""


class WarmingTargetTier(str, Enum):
    """Target cache tier for warming."""

    L1 = "l1"
    """L1 in-memory cache (Moka) - fastest, limited size."""

    L2 = "l2"
    """L2 local disk cache (RocksDB) - larger, persistent."""

    BOTH = "both"
    """Both L1 and L2 caches."""


class AccessPatternHint(str, Enum):
    """Access pattern hint for cache optimization."""

    RANDOM = "random"
    """Random access pattern."""

    SEQUENTIAL = "sequential"
    """Sequential access pattern."""

    TEMPORAL = "temporal"
    """Temporal locality (recently accessed items accessed again)."""

    SPATIAL = "spatial"
    """Spatial locality (nearby items accessed together)."""


@dataclass
class WarmCacheRequest:
    """Cache warming request with priority hints."""

    namespace: str
    vector_ids: list[str] | None = None
    priority: WarmingPriority = WarmingPriority.NORMAL
    target_tier: WarmingTargetTier = WarmingTargetTier.L2
    background: bool = False
    ttl_hint_seconds: int | None = None
    access_pattern: AccessPatternHint = AccessPatternHint.RANDOM
    max_vectors: int | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "namespace": self.namespace,
            "priority": self.priority.value,
            "target_tier": self.target_tier.value,
            "background": self.background,
            "access_pattern": self.access_pattern.value,
        }
        if self.vector_ids is not None:
            result["vector_ids"] = self.vector_ids
        if self.ttl_hint_seconds is not None:
            result["ttl_hint_seconds"] = self.ttl_hint_seconds
        if self.max_vectors is not None:
            result["max_vectors"] = self.max_vectors
        return result


@dataclass
class WarmCacheResponse:
    """Cache warming response."""

    success: bool
    entries_warmed: int
    entries_skipped: int
    message: str
    target_tier: WarmingTargetTier
    priority: WarmingPriority
    job_id: str | None = None
    estimated_completion: str | None = None
    bytes_warmed: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WarmCacheResponse":
        return cls(
            success=data["success"],
            entries_warmed=data.get("entries_warmed", 0),
            entries_skipped=data.get("entries_skipped", 0),
            message=data.get("message", ""),
            target_tier=WarmingTargetTier(data.get("target_tier", "l2")),
            priority=WarmingPriority(data.get("priority", "normal")),
            job_id=data.get("job_id"),
            estimated_completion=data.get("estimated_completion"),
            bytes_warmed=data.get("bytes_warmed"),
        )


# ============================================================================
# Namespace Configuration Types (v0.6.0)
# ============================================================================


@dataclass
class ConfigureNamespaceRequest:
    """Request body for ``PUT /v1/namespaces/:namespace`` (upsert semantics).

    Creates the namespace if it does not exist, or updates its configuration
    if it already exists.  Replaces the separate ``POST /v1/namespaces`` +
    future ``PATCH`` pattern with a single idempotent call.
    """

    dimension: int
    """Vector dimension.  Required on first creation; must match existing
    dimension on subsequent calls."""

    distance: DistanceMetric | None = None
    """Distance metric.  Defaults to ``cosine`` when not supplied."""

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"dimension": self.dimension}
        if self.distance is not None:
            result["distance"] = self.distance.value
        return result


@dataclass
class ConfigureNamespaceResponse:
    """Response from ``PUT /v1/namespaces/:namespace``."""

    namespace: str
    """Namespace name."""

    dimension: int
    """Vector dimension."""

    distance: DistanceMetric
    """Distance metric in use."""

    created: bool
    """``True`` if the namespace was newly created; ``False`` if it already existed."""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConfigureNamespaceResponse":
        return cls(
            namespace=data["namespace"],
            dimension=data["dimension"],
            distance=DistanceMetric(data.get("distance", "cosine")),
            created=data.get("created", False),
        )


# ============================================================================
# Vector Types
# ============================================================================


@dataclass
class Vector:
    """Represents a vector with optional metadata."""

    id: str
    values: list[float]
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API requests."""
        result: dict[str, Any] = {"id": self.id, "values": self.values}
        if self.metadata is not None:
            result["metadata"] = self.metadata
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Vector":
        """Create Vector from API response dictionary."""
        return cls(
            id=data["id"],
            values=data["values"],
            metadata=data.get("metadata"),
        )


@dataclass
class QueryResult:
    """Result from a vector query operation."""

    id: str
    score: float
    values: list[float] | None = None
    metadata: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QueryResult":
        """Create QueryResult from API response dictionary."""
        return cls(
            id=data["id"],
            score=data["score"],
            values=data.get("values"),
            metadata=data.get("metadata"),
        )


@dataclass
class SearchResult:
    """Result container for vector search operations."""

    results: list[QueryResult]
    total_searched: int | None = None
    search_time_ms: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SearchResult":
        """Create SearchResult from API response dictionary."""
        return cls(
            results=[QueryResult.from_dict(r) for r in data.get("results", [])],
            total_searched=data.get("total_searched"),
            search_time_ms=data.get("search_time_ms"),
        )


@dataclass
class NamespaceInfo:
    """Information about a namespace."""

    name: str
    vector_count: int
    dimensions: int | None = None
    index_type: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    metadata: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NamespaceInfo":
        """Create NamespaceInfo from API response dictionary."""
        return cls(
            name=data["name"],
            vector_count=data.get("vector_count", 0),
            dimensions=data.get("dimensions"),
            index_type=data.get("index_type"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            metadata=data.get("metadata"),
        )


@dataclass
class IndexStats:
    """Statistics about an index."""

    total_vectors: int
    dimensions: int
    index_type: str
    memory_usage_bytes: int | None = None
    disk_usage_bytes: int | None = None
    build_progress: float | None = None
    is_trained: bool | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IndexStats":
        """Create IndexStats from API response dictionary."""
        return cls(
            total_vectors=data.get("total_vectors", 0),
            dimensions=data.get("dimensions", 0),
            index_type=data.get("index_type", "unknown"),
            memory_usage_bytes=data.get("memory_usage_bytes"),
            disk_usage_bytes=data.get("disk_usage_bytes"),
            build_progress=data.get("build_progress"),
            is_trained=data.get("is_trained"),
        )


@dataclass
class Document:
    """Represents a document for full-text search."""

    id: str
    content: str
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API requests."""
        result: dict[str, Any] = {"id": self.id, "content": self.content}
        if self.metadata is not None:
            result["metadata"] = self.metadata
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Document":
        """Create Document from API response dictionary."""
        return cls(
            id=data["id"],
            content=data["content"],
            metadata=data.get("metadata"),
        )


@dataclass
class FullTextSearchResult:
    """Result from a full-text search operation."""

    id: str
    score: float
    content: str | None = None
    highlights: list[str] | None = None
    metadata: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FullTextSearchResult":
        """Create FullTextSearchResult from API response dictionary."""
        return cls(
            id=data["id"],
            score=data["score"],
            content=data.get("content"),
            highlights=data.get("highlights"),
            metadata=data.get("metadata"),
        )


@dataclass
class HybridSearchResult:
    """Result from a hybrid search operation."""

    id: str
    score: float
    vector_score: float | None = None
    text_score: float | None = None
    values: list[float] | None = None
    content: str | None = None
    metadata: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HybridSearchResult":
        """Create HybridSearchResult from API response dictionary."""
        return cls(
            id=data["id"],
            score=data["score"],
            vector_score=data.get("vector_score"),
            text_score=data.get("text_score"),
            values=data.get("values"),
            content=data.get("content"),
            metadata=data.get("metadata"),
        )


# ============================================================================
# Text-Based Inference Types (Auto-Embedding)
# ============================================================================


class EmbeddingModel(str, Enum):
    """
    Supported embedding models for text-based operations.

    - MINILM: MiniLM-L6 - Fast, good quality (384 dimensions)
    - BGE_SMALL: BGE-small - Balanced performance (384 dimensions)
    - E5_SMALL: E5-small - High quality (384 dimensions)
    """

    MINILM = "minilm"
    BGE_SMALL = "bge-small"
    E5_SMALL = "e5-small"


@dataclass
class TextDocument:
    """Input for upserting a text document with automatic embedding."""

    id: str
    """Unique identifier for the document."""

    text: str
    """Raw text content to be embedded."""

    metadata: dict[str, Any] | None = None
    """Optional metadata for the document."""

    ttl_seconds: int | None = None
    """Optional TTL in seconds."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API requests."""
        result: dict[str, Any] = {"id": self.id, "text": self.text}
        if self.metadata is not None:
            result["metadata"] = self.metadata
        if self.ttl_seconds is not None:
            result["ttl_seconds"] = self.ttl_seconds
        return result


@dataclass
class TextSearchResult:
    """A single text search result."""

    id: str
    """Document ID."""

    score: float
    """Similarity score."""

    text: str | None = None
    """Original text (if includeText was true)."""

    metadata: dict[str, Any] | None = None
    """Document metadata (excluding internal _text field)."""

    vector: list[float] | None = None
    """Vector values (if includeVectors was true)."""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TextSearchResult":
        """Create TextSearchResult from API response dictionary."""
        return cls(
            id=data["id"],
            score=data["score"],
            text=data.get("text"),
            metadata=data.get("metadata"),
            vector=data.get("vector"),
        )


@dataclass
class TextUpsertResponse:
    """Response from a text upsert operation."""

    upserted_count: int
    """Number of documents upserted."""

    tokens_processed: int
    """Approximate number of tokens processed."""

    model: EmbeddingModel
    """Embedding model used."""

    embedding_time_ms: int
    """Time spent generating embeddings in milliseconds."""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TextUpsertResponse":
        """Create TextUpsertResponse from API response dictionary."""
        return cls(
            upserted_count=data.get("upserted_count", 0),
            tokens_processed=data.get("tokens_processed", 0),
            model=EmbeddingModel(data.get("model", EmbeddingModel.MINILM.value)),
            embedding_time_ms=data.get("embedding_time_ms", 0),
        )


@dataclass
class TextQueryResponse:
    """Response from a text query operation."""

    results: list[TextSearchResult]
    """Search results."""

    model: EmbeddingModel
    """Embedding model used."""

    embedding_time_ms: int
    """Time spent generating query embedding in milliseconds."""

    search_time_ms: int
    """Time spent searching in milliseconds."""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TextQueryResponse":
        """Create TextQueryResponse from API response dictionary."""
        return cls(
            results=[TextSearchResult.from_dict(r) for r in data.get("results", [])],
            model=EmbeddingModel(data.get("model", EmbeddingModel.MINILM.value)),
            embedding_time_ms=data.get("embedding_time_ms", 0),
            search_time_ms=data.get("search_time_ms", 0),
        )


@dataclass
class BatchTextQueryResponse:
    """Response from a batch text query operation."""

    results: list[list[TextSearchResult]]
    """Results for each query."""

    model: EmbeddingModel
    """Embedding model used."""

    embedding_time_ms: int
    """Time spent generating all embeddings in milliseconds."""

    search_time_ms: int
    """Time spent on all searches in milliseconds."""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BatchTextQueryResponse":
        """Create BatchTextQueryResponse from API response dictionary."""
        return cls(
            results=[
                [TextSearchResult.from_dict(r) for r in query_results]
                for query_results in data.get("results", [])
            ],
            model=EmbeddingModel(data.get("model", EmbeddingModel.MINILM.value)),
            embedding_time_ms=data.get("embedding_time_ms", 0),
            search_time_ms=data.get("search_time_ms", 0),
        )


# ===========================================================================
# Memory Types
# ===========================================================================


@dataclass
class StoreMemoryRequest:
    """Request to store a memory."""
    content: str
    memory_type: str = "episodic"
    importance: float | None = None
    metadata: dict[str, Any] | None = None
    ttl_seconds: int | None = None
    expires_at: int | None = None
    session_id: str | None = None
    embedding: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"content": self.content, "memory_type": self.memory_type}
        if self.importance is not None:
            d["importance"] = self.importance
        if self.metadata is not None:
            d["metadata"] = self.metadata
        if self.ttl_seconds is not None:
            d["ttl_seconds"] = self.ttl_seconds
        if self.expires_at is not None:
            d["expires_at"] = self.expires_at
        if self.session_id is not None:
            d["session_id"] = self.session_id
        if self.embedding is not None:
            d["embedding"] = self.embedding
        return d


@dataclass
class Memory:
    """A stored memory."""
    id: str
    content: str
    memory_type: str
    importance: float
    metadata: dict[str, Any] | None = None
    created_at: str | None = None
    updated_at: str | None = None
    access_count: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Memory":
        return cls(
            id=data["id"],
            content=data["content"],
            memory_type=data.get("memory_type", "episodic"),
            importance=data.get("importance", 0.5),
            metadata=data.get("metadata"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            access_count=data.get("access_count"),
        )


@dataclass
class RecalledMemory:
    """A recalled memory with similarity score."""
    id: str
    content: str
    memory_type: str
    importance: float
    score: float
    metadata: dict[str, Any] | None = None
    created_at: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RecalledMemory":
        return cls(
            id=data["id"],
            content=data["content"],
            memory_type=data.get("memory_type", "episodic"),
            importance=data.get("importance", 0.5),
            score=data.get("score", 0.0),
            metadata=data.get("metadata"),
            created_at=data.get("created_at"),
        )


@dataclass
class ConsolidateResponse:
    """Response from memory consolidation."""
    consolidated_count: int
    removed_count: int
    new_memories: list[str]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConsolidateResponse":
        return cls(
            consolidated_count=data.get("consolidated_count", 0),
            removed_count=data.get("removed_count", 0),
            new_memories=data.get("new_memories", []),
        )


# ===========================================================================
# Session Types
# ===========================================================================


@dataclass
class Session:
    """A session."""
    session_id: str
    agent_id: str
    started_at: str | None = None
    ended_at: str | None = None
    metadata: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Session":
        return cls(
            session_id=data["session_id"],
            agent_id=data["agent_id"],
            started_at=data.get("started_at"),
            ended_at=data.get("ended_at"),
            metadata=data.get("metadata"),
        )


# ===========================================================================
# Agent Types
# ===========================================================================


@dataclass
class AgentSummary:
    """Summary info for an agent."""
    agent_id: str
    memory_count: int
    session_count: int
    active_sessions: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentSummary":
        return cls(
            agent_id=data["agent_id"],
            memory_count=data.get("memory_count", 0),
            session_count=data.get("session_count", 0),
            active_sessions=data.get("active_sessions", 0),
        )


@dataclass
class AgentStats:
    """Detailed stats for an agent."""
    agent_id: str
    total_memories: int
    memories_by_type: dict[str, int]
    total_sessions: int
    active_sessions: int
    avg_importance: float | None = None
    oldest_memory_at: str | None = None
    newest_memory_at: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentStats":
        return cls(
            agent_id=data["agent_id"],
            total_memories=data.get("total_memories", 0),
            memories_by_type=data.get("memories_by_type", {}),
            total_sessions=data.get("total_sessions", 0),
            active_sessions=data.get("active_sessions", 0),
            avg_importance=data.get("avg_importance"),
            oldest_memory_at=data.get("oldest_memory_at"),
            newest_memory_at=data.get("newest_memory_at"),
        )


# ===========================================================================
# Knowledge Graph Types
# ===========================================================================


@dataclass
class KnowledgeNode:
    """A node in the knowledge graph."""
    id: str
    content: str
    memory_type: str | None = None
    importance: float | None = None
    metadata: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KnowledgeNode":
        return cls(
            id=data["id"],
            content=data["content"],
            memory_type=data.get("memory_type"),
            importance=data.get("importance"),
            metadata=data.get("metadata"),
        )


@dataclass
class KnowledgeEdge:
    """An edge in the knowledge graph."""
    source: str
    target: str
    similarity: float
    relationship: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KnowledgeEdge":
        return cls(
            source=data["source"],
            target=data["target"],
            similarity=data.get("similarity", 0.0),
            relationship=data.get("relationship"),
        )


@dataclass
class KnowledgeGraphResponse:
    """Response from knowledge graph operations."""
    nodes: list["KnowledgeNode"]
    edges: list["KnowledgeEdge"]
    clusters: list[list[str]] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KnowledgeGraphResponse":
        return cls(
            nodes=[KnowledgeNode.from_dict(n) for n in data.get("nodes", [])],
            edges=[KnowledgeEdge.from_dict(e) for e in data.get("edges", [])],
            clusters=data.get("clusters"),
        )


@dataclass
class SummarizeResponse:
    """Response from summarization."""
    summary: str
    source_count: int
    new_memory_id: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SummarizeResponse":
        return cls(
            summary=data["summary"],
            source_count=data.get("source_count", 0),
            new_memory_id=data.get("new_memory_id"),
        )


@dataclass
class DeduplicateResponse:
    """Response from deduplication."""
    duplicates_found: int
    removed_count: int
    groups: list[list[str]]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DeduplicateResponse":
        return cls(
            duplicates_found=data.get("duplicates_found", 0),
            removed_count=data.get("removed_count", 0),
            groups=data.get("groups", []),
        )


# ===========================================================================
# Analytics Types
# ===========================================================================


@dataclass
class AnalyticsOverview:
    """Analytics overview response."""
    total_queries: int
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    queries_per_second: float
    error_rate: float
    cache_hit_rate: float
    storage_used_bytes: int
    total_vectors: int
    total_namespaces: int
    uptime_seconds: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnalyticsOverview":
        return cls(
            total_queries=data.get("total_queries", 0),
            avg_latency_ms=data.get("avg_latency_ms", 0.0),
            p95_latency_ms=data.get("p95_latency_ms", 0.0),
            p99_latency_ms=data.get("p99_latency_ms", 0.0),
            queries_per_second=data.get("queries_per_second", 0.0),
            error_rate=data.get("error_rate", 0.0),
            cache_hit_rate=data.get("cache_hit_rate", 0.0),
            storage_used_bytes=data.get("storage_used_bytes", 0),
            total_vectors=data.get("total_vectors", 0),
            total_namespaces=data.get("total_namespaces", 0),
            uptime_seconds=data.get("uptime_seconds", 0),
        )


# Type aliases for convenience
FilterDict = dict[str, Any]
MetadataDict = dict[str, Any]
VectorValues = list[float]
VectorInput = Union[Vector, dict[str, Any]]
DocumentInput = Union[Document, dict[str, Any]]
TextDocumentInput = Union[TextDocument, dict[str, Any]]


# ===========================================================================
# SSE Streaming Event Types (CE-1)
# ===========================================================================


class OpStatus(str, Enum):
    """Operation status for ``operation_progress`` events."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class VectorMutationOp(str, Enum):
    """Vector mutation operation type for ``vectors_mutated`` events."""

    UPSERTED = "upserted"
    DELETED = "deleted"


@dataclass
class DakeraEvent:
    """An event received from a Dakera SSE stream.

    The ``type`` field identifies the event variant.  All other fields are
    optional and populated based on the specific event type:

    - ``namespace_created``: namespace, dimension
    - ``namespace_deleted``: namespace
    - ``operation_progress``: operation_id, namespace, op_type, progress,
      status, message, updated_at
    - ``job_progress``: job_id, job_type, namespace, progress, status
    - ``vectors_mutated``: namespace, op, count
    - ``stream_lagged``: dropped, hint (reconnect to resume)

    Example::

        for event in client.stream_namespace_events("my-ns"):
            if event.type == "vectors_mutated":
                print(f"{event.count} vectors {event.op} in {event.namespace}")
    """

    type: str
    # namespace_created / namespace_deleted / vectors_mutated / operation_progress
    namespace: str | None = None
    # namespace_created
    dimension: int | None = None
    # operation_progress
    operation_id: str | None = None
    op_type: str | None = None
    progress: int | None = None
    status: str | None = None
    message: str | None = None
    updated_at: int | None = None
    # job_progress
    job_id: str | None = None
    job_type: str | None = None
    # vectors_mutated
    op: str | None = None
    count: int | None = None
    # stream_lagged
    dropped: int | None = None
    hint: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DakeraEvent":
        """Create a :class:`DakeraEvent` from a parsed SSE data payload."""
        import dataclasses

        valid = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid})


# ===========================================================================
# DASH-B: Memory Lifecycle Event Types
# ===========================================================================


@dataclass
class MemoryEvent:
    """A memory lifecycle event from the DASH-B SSE stream.

    Received from ``GET /v1/events/stream`` (Read scope).

    The ``event_type`` field identifies the operation:

    - ``connected`` — emitted immediately on subscription (agent_id will be empty)
    - ``stored`` — a memory was stored (content, importance, tags present)
    - ``recalled`` — a memory was recalled (importance present)
    - ``forgotten`` — a memory was deleted
    - ``consolidated`` — memories were merged (memory_id is the new memory)
    - ``importance_updated`` — importance score changed
    - ``session_started`` — an agent session began (session_id present)
    - ``session_ended`` — an agent session ended (session_id present)
    - ``stream_lagged`` — consumer fell behind; some events were dropped

    Example::

        for event in client.stream_memory_events():
            if event.event_type == "stored":
                print(f"[{event.agent_id}] stored {event.memory_id}")
    """

    event_type: str
    timestamp: int
    agent_id: str = ""
    memory_id: str | None = None
    content: str | None = None
    importance: float | None = None
    tags: list[str] | None = None
    session_id: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryEvent":
        """Create a :class:`MemoryEvent` from a parsed SSE data payload.

        Handles both regular memory events (``event_type`` field) and the
        ``connected`` handshake event (``type`` field, no ``agent_id``).
        """
        import dataclasses

        valid = {f.name for f in dataclasses.fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in valid}
        # connected event uses "type" instead of "event_type"
        if "event_type" not in kwargs and "type" in data:
            kwargs["event_type"] = data["type"]
        return cls(**kwargs)


# ===========================================================================
# DASH-A: Cross-Agent Network Types
# ===========================================================================


@dataclass
class AgentNetworkInfo:
    """Summary information for one agent in the cross-agent network."""

    agent_id: str
    memory_count: int
    avg_importance: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentNetworkInfo":
        return cls(
            agent_id=data["agent_id"],
            memory_count=data["memory_count"],
            avg_importance=data["avg_importance"],
        )


@dataclass
class AgentNetworkNode:
    """A memory node in the cross-agent network graph."""

    id: str
    agent_id: str
    content: str
    importance: float
    tags: list[str]
    memory_type: str
    created_at: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentNetworkNode":
        return cls(
            id=data["id"],
            agent_id=data["agent_id"],
            content=data["content"],
            importance=data["importance"],
            tags=data.get("tags", []),
            memory_type=data["memory_type"],
            created_at=data["created_at"],
        )


@dataclass
class AgentNetworkEdge:
    """A similarity edge between memories from two different agents."""

    source: str
    target: str
    source_agent: str
    target_agent: str
    similarity: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentNetworkEdge":
        return cls(
            source=data["source"],
            target=data["target"],
            source_agent=data["source_agent"],
            target_agent=data["target_agent"],
            similarity=data["similarity"],
        )


@dataclass
class AgentNetworkStats:
    """Network-level statistics for the cross-agent graph."""

    total_agents: int
    total_nodes: int
    total_cross_edges: int
    density: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentNetworkStats":
        return cls(
            total_agents=data["total_agents"],
            total_nodes=data["total_nodes"],
            total_cross_edges=data["total_cross_edges"],
            density=data["density"],
        )


@dataclass
class CrossAgentNetworkResponse:
    """Response from ``POST /v1/knowledge/network/cross-agent``.

    Contains agents, memory nodes, inter-agent similarity edges, and
    aggregate network statistics.
    """

    agents: list[AgentNetworkInfo]
    nodes: list[AgentNetworkNode]
    edges: list[AgentNetworkEdge]
    stats: AgentNetworkStats
    node_count: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CrossAgentNetworkResponse":
        return cls(
            agents=[AgentNetworkInfo.from_dict(a) for a in data.get("agents", [])],
            nodes=[AgentNetworkNode.from_dict(n) for n in data.get("nodes", [])],
            edges=[AgentNetworkEdge.from_dict(e) for e in data.get("edges", [])],
            stats=AgentNetworkStats.from_dict(data["stats"]),
            node_count=int(data.get("node_count", 0)),
        )


# ===========================================================================
# OPS-1: Rate-Limit Headers
# ===========================================================================


@dataclass
class RateLimitHeaders:
    """Rate-limit and quota headers returned on every API response (OPS-1).

    Fields are ``None`` when the server does not include the header
    (e.g. non-namespaced endpoints where quota does not apply).
    """

    limit: int | None = None
    """``X-RateLimit-Limit`` — max requests allowed in the current window."""
    remaining: int | None = None
    """``X-RateLimit-Remaining`` — requests left in the current window."""
    reset: int | None = None
    """``X-RateLimit-Reset`` — Unix timestamp (seconds) when the window resets."""
    quota_used: int | None = None
    """``X-Quota-Used`` — namespace vectors / storage consumed."""
    quota_limit: int | None = None
    """``X-Quota-Limit`` — namespace quota ceiling."""

    @classmethod
    def from_headers(cls, headers: dict[str, str]) -> "RateLimitHeaders":
        def _int(key: str) -> int | None:
            val = headers.get(key)
            try:
                return int(val) if val is not None else None
            except (ValueError, TypeError):
                return None

        return cls(
            limit=_int("X-RateLimit-Limit"),
            remaining=_int("X-RateLimit-Remaining"),
            reset=_int("X-RateLimit-Reset"),
            quota_used=_int("X-Quota-Used"),
            quota_limit=_int("X-Quota-Limit"),
        )


# ===========================================================================
# CE-2: Batch Recall / Forget
# ===========================================================================


@dataclass
class BatchMemoryFilter:
    """Filter predicates for batch memory operations (CE-2).

    All fields are optional.  For ``batch_forget`` at least one must be set
    (server-side safety guard).
    """

    tags: list[str] | None = None
    """Restrict to memories that carry **all** listed tags."""
    min_importance: float | None = None
    """Minimum importance (inclusive)."""
    max_importance: float | None = None
    """Maximum importance (inclusive)."""
    created_after: int | None = None
    """Only memories created at or after this Unix timestamp (seconds)."""
    created_before: int | None = None
    """Only memories created before or at this Unix timestamp (seconds)."""
    memory_type: str | None = None
    """Restrict to a specific memory type (e.g. ``"episodic"``)."""
    session_id: str | None = None
    """Restrict to memories from a specific session."""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        if self.tags is not None:
            d["tags"] = self.tags
        if self.min_importance is not None:
            d["min_importance"] = self.min_importance
        if self.max_importance is not None:
            d["max_importance"] = self.max_importance
        if self.created_after is not None:
            d["created_after"] = self.created_after
        if self.created_before is not None:
            d["created_before"] = self.created_before
        if self.memory_type is not None:
            d["memory_type"] = self.memory_type
        if self.session_id is not None:
            d["session_id"] = self.session_id
        return d


@dataclass
class BatchRecallRequest:
    """Request body for ``POST /v1/memories/recall/batch``."""

    agent_id: str
    """Agent whose memory namespace to search."""
    filter: BatchMemoryFilter | None = None
    """Filter predicates to apply.  An empty filter returns all memories up to ``limit``."""
    limit: int = 100
    """Maximum number of results to return (default: 100)."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "filter": self.filter.to_dict() if self.filter else {},
            "limit": self.limit,
        }


@dataclass
class BatchRecallResponse:
    """Response from ``POST /v1/memories/recall/batch``."""

    memories: list[Memory]
    total: int
    filtered: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BatchRecallResponse":
        return cls(
            memories=[Memory.from_dict(m) for m in data.get("memories", [])],
            total=data.get("total", 0),
            filtered=data.get("filtered", 0),
        )


@dataclass
class BatchForgetRequest:
    """Request body for ``DELETE /v1/memories/forget/batch``."""

    agent_id: str
    """Agent whose memory namespace to purge from."""
    filter: BatchMemoryFilter = None  # type: ignore[assignment]
    """Filter predicates — **at least one must be set** (server safety guard)."""

    def __post_init__(self) -> None:
        if self.filter is None:
            self.filter = BatchMemoryFilter()

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "filter": self.filter.to_dict(),
        }


@dataclass
class BatchForgetResponse:
    """Response from ``DELETE /v1/memories/forget/batch``."""

    deleted_count: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BatchForgetResponse":
        return cls(deleted_count=data.get("deleted_count", 0))


# ============================================================================
# Memory Knowledge Graph Types (CE-5 / SDK-9)
# ============================================================================


class EdgeType(str, Enum):
    """Edge type for memory knowledge graph relationships (CE-5).

    - ``related_to``: Cosine similarity ≥ 0.85 — two memories are semantically similar.
    - ``shares_entity``: Both memories reference the same named entity (CE-4 tags).
    - ``precedes``: Temporal ordering — one memory was created before the other.
    - ``linked_by``: Explicit user/agent-created link via ``POST /v1/memories/{id}/links``.
    """

    RELATED_TO = "related_to"
    SHARES_ENTITY = "shares_entity"
    PRECEDES = "precedes"
    LINKED_BY = "linked_by"


@dataclass
class GraphEdge:
    """A directed edge in the memory knowledge graph."""

    id: str
    """Unique edge identifier."""
    source_id: str
    """Source memory ID."""
    target_id: str
    """Target memory ID."""
    edge_type: EdgeType
    """Relationship type between the two memories."""
    weight: float
    """Edge weight (0.0–1.0). For ``related_to`` this is the cosine similarity score."""
    created_at: int
    """Unix timestamp of edge creation."""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GraphEdge":
        return cls(
            id=data["id"],
            source_id=data["source_id"],
            target_id=data["target_id"],
            edge_type=EdgeType(data["edge_type"]),
            weight=data.get("weight", 0.0),
            created_at=data.get("created_at", 0),
        )


@dataclass
class GraphNode:
    """A node (memory) in the knowledge graph traversal result."""

    memory_id: str
    """Memory identifier."""
    content_preview: str
    """First 200 characters of memory content."""
    importance: float
    """Memory importance score."""
    depth: int
    """Traversal depth from the root node (root = 0)."""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GraphNode":
        return cls(
            memory_id=data["memory_id"],
            content_preview=data.get("content_preview", ""),
            importance=data.get("importance", 0.0),
            depth=data.get("depth", 0),
        )


@dataclass
class MemoryGraph:
    """Graph traversal result from ``GET /v1/memories/{id}/graph``."""

    root_id: str
    """The root memory ID from which traversal started."""
    depth: int
    """Maximum traversal depth used."""
    nodes: list[GraphNode]
    """All memory nodes reachable within the requested depth."""
    edges: list[GraphEdge]
    """All edges connecting the returned nodes."""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryGraph":
        return cls(
            root_id=data["root_id"],
            depth=data.get("depth", 1),
            nodes=[GraphNode.from_dict(n) for n in data.get("nodes", [])],
            edges=[GraphEdge.from_dict(e) for e in data.get("edges", [])],
        )


@dataclass
class GraphPath:
    """Shortest path between two memories from ``GET /v1/memories/{id}/path``."""

    source_id: str
    """Starting memory ID."""
    target_id: str
    """Destination memory ID."""
    path: list[str]
    """Ordered list of memory IDs from source to target (inclusive)."""
    hops: int
    """Number of edges traversed (``len(path) - 1``)."""
    edges: list[GraphEdge]
    """Edges along the path, in traversal order."""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GraphPath":
        edges = [GraphEdge.from_dict(e) for e in data.get("edges", [])]
        path: list[str] = data.get("path", [])
        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            path=path,
            hops=data.get("hops", max(0, len(path) - 1)),
            edges=edges,
        )


@dataclass
class GraphLinkResponse:
    """Response from ``POST /v1/memories/{id}/links``."""

    edge: GraphEdge
    """The newly created edge."""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GraphLinkResponse":
        return cls(edge=GraphEdge.from_dict(data["edge"]))


@dataclass
class GraphExport:
    """Agent graph export from ``GET /v1/agents/{id}/graph/export``."""

    agent_id: str
    """Agent whose graph was exported."""
    format: str
    """Export format: ``json``, ``graphml``, or ``csv``."""
    data: str
    """Serialised graph in the requested format."""
    node_count: int
    """Total number of memory nodes in the export."""
    edge_count: int
    """Total number of edges in the export."""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GraphExport":
        return cls(
            agent_id=data["agent_id"],
            format=data["format"],
            data=data["data"],
            node_count=data.get("node_count", 0),
            edge_count=data.get("edge_count", 0),
        )


# ============================================================================
# Entity Extraction Types (CE-4 / GLiNER)
# ============================================================================


@dataclass
class NamespaceNerConfig:
    """Entity extraction configuration for a namespace."""
    extract_entities: bool = False
    entity_types: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"extract_entities": self.extract_entities}
        if self.entity_types is not None:
            d["entity_types"] = self.entity_types
        return d

@dataclass
class ExtractedEntity:
    """A single extracted entity."""
    entity_type: str
    value: str
    score: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExtractedEntity":
        return cls(
            entity_type=data["entity_type"],
            value=data["value"],
            score=float(data.get("score", 0.0)),
        )

@dataclass
class EntityExtractionResponse:
    """Response from POST /v1/memories/extract."""
    entities: list[ExtractedEntity]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EntityExtractionResponse":
        return cls(
            entities=[ExtractedEntity.from_dict(e) for e in data.get("entities", [])],
        )

@dataclass
class MemoryEntitiesResponse:
    """Response from GET /v1/memory/entities/:id."""
    memory_id: str
    entities: list[ExtractedEntity]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryEntitiesResponse":
        return cls(
            memory_id=data["memory_id"],
            entities=[ExtractedEntity.from_dict(e) for e in data.get("entities", [])],
        )
