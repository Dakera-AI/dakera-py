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


class RoutingMode(str, Enum):
    """Routing mode for recall and search (CE-10).

    Controls which retrieval index to use when recalling or searching memories.
    ``auto`` (default) lets the server pick the best strategy based on the query.
    """

    AUTO = "auto"
    """Server picks the best strategy (default)."""
    VECTOR = "vector"
    """Force ANN vector search (HNSW)."""
    BM25 = "bm25"
    """Force BM25 full-text search."""
    HYBRID = "hybrid"
    """Fuse ANN and BM25 scores (RRF)."""


class FusionStrategy(str, Enum):
    """Fusion strategy for hybrid recall (CE-14).

    Controls how vector and BM25 scores are combined when ``routing=hybrid``.
    ``rrf`` (default) uses Reciprocal Rank Fusion (Cormack et al., SIGIR 2009)
    which is rank-based and scale-invariant. ``minmax`` uses legacy weighted
    min-max normalization.
    """

    RRF = "rrf"
    """Reciprocal Rank Fusion — default, best for recall tasks."""
    MINMAX = "minmax"
    """Weighted min-max normalization — legacy behavior."""


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
            name=data.get("name") or data.get("namespace", ""),
            vector_count=data.get("vector_count", 0),
            dimensions=data.get("dimension") or data.get("dimensions"),
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
    text: str
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API requests."""
        result: dict[str, Any] = {"id": self.id, "text": self.text}
        if self.metadata is not None:
            result["metadata"] = self.metadata
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Document":
        """Create Document from API response dictionary."""
        return cls(
            id=data["id"],
            text=data["text"],
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

    - BGE_LARGE: BGE-large - Best quality, default (1024 dimensions)
    - MINILM: MiniLM-L6 - Fast, good quality (384 dimensions)
    - BGE_SMALL: BGE-small - Balanced performance (384 dimensions)
    - E5_SMALL: E5-small - High quality (384 dimensions)
    - MODERNBERT_EMBED_BASE: ModernBERT-embed-base - 768 dimensions, MRL, 8192 tokens
    - GTE_MODERNBERT_BASE: GTE-ModernBERT-base - 768 dimensions, MTEB retrieval 64.38
    """

    BGE_LARGE = "bge-large"
    MINILM = "minilm"
    BGE_SMALL = "bge-small"
    E5_SMALL = "e5-small"
    MODERNBERT_EMBED_BASE = "modernbert-embed-base"
    GTE_MODERNBERT_BASE = "gte-modernbert-base"


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
    """Ranking score — equals smart_score when present, then weighted_score, then raw score."""
    smart_score: float | None = None
    """Raw smart_score from the server (the primary ranking key)."""
    weighted_score: float | None = None
    """Raw weighted_score from the server."""
    metadata: dict[str, Any] | None = None
    created_at: str | None = None
    depth: int | None = None
    """KG-3: hop depth at which this memory was found (only set on associated memories)."""
    vector_score: float | None = None
    """Hybrid sub-score: vector similarity component (server v0.11.98+, absent when BM25-only)."""
    text_score: float | None = None
    """Hybrid sub-score: BM25 text component (server v0.11.98+, absent when vector-only)."""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RecalledMemory":
        smart_score = data.get("smart_score")
        weighted_score = data.get("weighted_score")
        if smart_score is not None:
            score = smart_score
        elif weighted_score is not None:
            score = weighted_score
        else:
            score = data.get("score", 0.0)
        return cls(
            id=data["id"],
            content=data["content"],
            memory_type=data.get("memory_type", "episodic"),
            importance=data.get("importance", 0.5),
            score=score,
            smart_score=smart_score,
            weighted_score=weighted_score,
            metadata=data.get("metadata"),
            created_at=data.get("created_at"),
            depth=data.get("depth"),
            vector_score=data.get("vector_score"),
            text_score=data.get("text_score"),
        )


@dataclass
class RecallResponse:
    """Response from the recall endpoint (COG-2 / KG-3).

    Contains primary recalled memories plus optional associatively linked
    memories surfaced via configurable-depth KG traversal when
    ``include_associated`` is requested.  Each associated memory carries a
    ``depth`` field indicating the hop at which it was found (KG-3).
    """

    memories: list["RecalledMemory"]
    associated_memories: list["RecalledMemory"] | None = None

    @classmethod
    def _normalize_memory(cls, m: dict[str, Any]) -> dict[str, Any]:
        """Flatten nested memory envelope; promote sub-score fields to top level."""
        if "memory" in m and isinstance(m["memory"], dict):
            flat = {**m["memory"], "score": m.get("score", 0.0)}
            for key in ("smart_score", "weighted_score", "vector_score", "text_score"):
                if key in m:
                    flat[key] = m[key]
            return flat
        return m

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RecallResponse":
        memories = [
            RecalledMemory.from_dict(cls._normalize_memory(m))
            for m in data.get("memories", [])
        ]
        raw_assoc = data.get("associated_memories")
        associated_memories = (
            [RecalledMemory.from_dict(cls._normalize_memory(m)) for m in raw_assoc]
            if raw_assoc is not None
            else None
        )
        return cls(memories=memories, associated_memories=associated_memories)


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


@dataclass
class WakeUpResponse:
    """Response from ``GET /v1/agents/{agent_id}/wake-up`` (DAK-1690).

    Returns top-N memories ranked by ``importance × exp(-ln2 × age / 14d)``
    for fast agent start-up context loading. No embedding inference — served
    from metadata index for sub-millisecond latency.

    Requires Read scope on the agent namespace.
    """

    agent_id: str
    """The agent whose memories are returned."""
    memories: list["Memory"]
    """Top-N memories ranked by recency-weighted importance."""
    total_available: int
    """Total memories available before top_n cap was applied."""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WakeUpResponse":
        return cls(
            agent_id=data["agent_id"],
            memories=[Memory.from_dict(m) for m in data.get("memories", [])],
            total_available=data.get("total_available", 0),
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


@dataclass
class CompressResponse:
    """Response from ``POST /v1/agents/{id}/compress`` (CE-12).

    Contains compression statistics for the agent's memory namespace after
    the server runs the compression pass.
    """

    agent_id: str
    """The agent whose namespace was compressed."""
    memories_before: int
    """Number of memories before compression."""
    memories_after: int
    """Number of memories after compression."""
    removed_count: int
    """Number of memories removed during compression."""
    duration_ms: float = 0.0
    """Wall-clock duration of the compression pass in milliseconds."""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CompressResponse":
        return cls(
            agent_id=data.get("agent_id", ""),
            memories_before=data.get("memories_before", 0),
            memories_after=data.get("memories_after", 0),
            removed_count=data.get("removed_count", 0),
            duration_ms=float(data.get("duration_ms", 0.0)),
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


class F:
    """Typed filter builder helpers — produce filter dicts matching the server DSL.

    Usage::

        # Equality / comparison
        {"importance": F.gte(0.8)}
        {"tags": F.array_contains("entity:PERSON:alice")}

        # Logical combinators
        F.and_({"importance": F.gte(0.8)}, {"tags": F.array_contains("entity:PERSON:alice")})
    """

    # ------------------------------------------------------------------
    # Comparison operators
    # ------------------------------------------------------------------
    @staticmethod
    def eq(value: Any) -> FilterDict:
        return {"$eq": value}

    @staticmethod
    def ne(value: Any) -> FilterDict:
        return {"$ne": value}

    @staticmethod
    def gt(value: Any) -> FilterDict:
        return {"$gt": value}

    @staticmethod
    def gte(value: Any) -> FilterDict:
        return {"$gte": value}

    @staticmethod
    def lt(value: Any) -> FilterDict:
        return {"$lt": value}

    @staticmethod
    def lte(value: Any) -> FilterDict:
        return {"$lte": value}

    @staticmethod
    def in_(values: list[Any]) -> FilterDict:
        return {"$in": values}

    @staticmethod
    def nin(values: list[Any]) -> FilterDict:
        return {"$nin": values}

    @staticmethod
    def exists(present: bool = True) -> FilterDict:
        return {"$exists": present}

    # ------------------------------------------------------------------
    # String operators
    # ------------------------------------------------------------------
    @staticmethod
    def contains(substr: str) -> FilterDict:
        """Case-sensitive substring match."""
        return {"$contains": substr}

    @staticmethod
    def icontains(substr: str) -> FilterDict:
        """Case-insensitive substring match."""
        return {"$icontains": substr}

    @staticmethod
    def starts_with(prefix: str) -> FilterDict:
        return {"$startsWith": prefix}

    @staticmethod
    def ends_with(suffix: str) -> FilterDict:
        return {"$endsWith": suffix}

    @staticmethod
    def glob(pattern: str) -> FilterDict:
        """Glob pattern match (supports * and ? wildcards)."""
        return {"$glob": pattern}

    @staticmethod
    def regex(pattern: str) -> FilterDict:
        return {"$regex": pattern}

    # ------------------------------------------------------------------
    # Array operators (CE-79)
    # ------------------------------------------------------------------
    @staticmethod
    def array_contains(value: Any) -> FilterDict:
        """Matches when an array metadata field contains *value*.

        Example — find memories tagged for a specific person::

            filter = {"tags": F.array_contains("entity:PERSON:alice")}
        """
        return {"$arrayContains": value}

    @staticmethod
    def array_contains_all(values: list[Any]) -> FilterDict:
        """Matches when an array field contains *all* of *values*."""
        return {"$arrayContainsAll": values}

    @staticmethod
    def array_contains_any(values: list[Any]) -> FilterDict:
        """Matches when an array field contains *any* of *values*."""
        return {"$arrayContainsAny": values}

    # ------------------------------------------------------------------
    # Logical combinators
    # ------------------------------------------------------------------
    @staticmethod
    def and_(*conditions: FilterDict) -> FilterDict:
        return {"$and": list(conditions)}

    @staticmethod
    def or_(*conditions: FilterDict) -> FilterDict:
        return {"$or": list(conditions)}


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
    min_importance: float | None = None
    """Convenience shortcut — sets ``filter.min_importance`` if no explicit filter is given."""

    def __post_init__(self) -> None:
        if self.min_importance is not None and self.filter is None:
            self.filter = BatchMemoryFilter(min_importance=self.min_importance)

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


@dataclass
class BatchStoreMemoryItem:
    """A single memory entry within a :class:`BatchStoreMemoryRequest` (DAK-5508).

    Mirrors :class:`StoreMemoryRequest` but omits ``agent_id`` — supplied at batch level.
    """

    content: str
    """Memory content text (required, max 100 000 chars)."""
    memory_type: str = "episodic"
    """One of ``"episodic"``, ``"semantic"``, ``"procedural"``, or ``"working"``."""
    importance: float = 0.5
    """Importance score 0.0–1.0 (default: 0.5)."""
    tags: list[str] | None = None
    """Optional tags to associate with the memory."""
    session_id: str | None = None
    """Optional session ID to associate with."""
    metadata: dict[str, Any] | None = None
    """Arbitrary metadata dictionary."""
    ttl_seconds: int | None = None
    """Optional TTL in seconds."""
    expires_at: int | None = None
    """Optional explicit expiry as a Unix timestamp (seconds)."""
    id: str | None = None
    """Optional custom ID. Auto-generated if not provided."""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "content": self.content,
            "memory_type": self.memory_type,
            "importance": self.importance,
        }
        if self.tags is not None:
            d["tags"] = self.tags
        if self.session_id is not None:
            d["session_id"] = self.session_id
        if self.metadata is not None:
            d["metadata"] = self.metadata
        if self.ttl_seconds is not None:
            d["ttl_seconds"] = self.ttl_seconds
        if self.expires_at is not None:
            d["expires_at"] = self.expires_at
        if self.id is not None:
            d["id"] = self.id
        return d


@dataclass
class BatchStoreMemoryRequest:
    """Request body for ``POST /v1/memories/store/batch`` (DAK-5508).

    Accepts up to 1 000 memories per call. The server embeds all contents in a
    single ONNX inference pass and upserts them in one write, yielding ≥100×
    throughput vs. N sequential single-store calls.
    """

    agent_id: str
    """Agent namespace to store the memories in."""
    memories: list[BatchStoreMemoryItem]
    """Memories to store (1–1000 items)."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "memories": [m.to_dict() for m in self.memories],
        }


@dataclass
class BatchStoredMemory:
    """A single stored memory returned in a :class:`BatchStoreMemoryResponse`."""

    id: str
    content: str
    agent_id: str
    tags: list[str]
    importance: float
    created_at: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BatchStoredMemory":
        return cls(
            id=data["id"],
            content=data["content"],
            agent_id=data["agent_id"],
            tags=data.get("tags", []),
            importance=data.get("importance", 0.5),
            created_at=data.get("created_at", 0),
        )


@dataclass
class BatchStoreMemoryResponse:
    """Response from ``POST /v1/memories/store/batch``."""

    stored: list[BatchStoredMemory]
    """Stored memories in the same order as the request items."""
    stored_count: int
    """Number of memories successfully stored."""
    total_embedding_time_ms: int
    """Time spent on ONNX embedding for the entire batch (milliseconds)."""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BatchStoreMemoryResponse":
        return cls(
            stored=[BatchStoredMemory.from_dict(m) for m in data.get("stored", [])],
            stored_count=data.get("stored_count", 0),
            total_embedding_time_ms=data.get("total_embedding_time_ms", 0),
        )


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
# KG-2: Graph Query & Export Types
# ============================================================================


@dataclass
class KgQueryResponse:
    """Response from ``GET /v1/knowledge/query`` (KG-2).

    Returned by :meth:`DakeraClient.knowledge_query` and
    :meth:`AsyncDakeraClient.knowledge_query`.
    """

    agent_id: str
    """Agent whose graph was queried."""
    node_count: int
    """Number of unique memory node IDs referenced by the returned edges."""
    edge_count: int
    """Number of edges returned."""
    edges: list[GraphEdge]
    """Matching edges, up to *limit*."""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KgQueryResponse":
        return cls(
            agent_id=data["agent_id"],
            node_count=data.get("node_count", 0),
            edge_count=data.get("edge_count", 0),
            edges=[GraphEdge.from_dict(e) for e in data.get("edges", [])],
        )


@dataclass
class KgPathResponse:
    """Response from ``GET /v1/knowledge/path`` (KG-2).

    Returned by :meth:`DakeraClient.knowledge_path` and
    :meth:`AsyncDakeraClient.knowledge_path`.
    """

    agent_id: str
    """Agent whose graph was traversed."""
    from_id: str
    """Source memory ID."""
    to_id: str
    """Target memory ID."""
    hop_count: int
    """Number of edges in the shortest path (0 if source == target)."""
    path: list[str]
    """Ordered list of memory IDs from source to target (inclusive)."""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KgPathResponse":
        return cls(
            agent_id=data["agent_id"],
            from_id=data["from_id"],
            to_id=data["to_id"],
            hop_count=data.get("hop_count", 0),
            path=data.get("path", []),
        )


@dataclass
class KgExportResponse:
    """Response from ``GET /v1/knowledge/export`` with ``format=json`` (KG-2).

    Returned by :meth:`DakeraClient.knowledge_export` and
    :meth:`AsyncDakeraClient.knowledge_export` when *format* is ``"json"``.
    For GraphML, the server returns ``application/xml`` — call the endpoint
    directly if you need the raw XML bytes.
    """

    agent_id: str
    """Agent whose graph was exported."""
    format: str
    """Export format used (``"json"`` when this dataclass is returned)."""
    node_count: int
    """Total number of unique memory node IDs in the export."""
    edge_count: int
    """Total number of edges in the export."""
    edges: list[GraphEdge]
    """All graph edges for the agent."""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KgExportResponse":
        return cls(
            agent_id=data["agent_id"],
            format=data.get("format", "json"),
            node_count=data.get("node_count", 0),
            edge_count=data.get("edge_count", 0),
            edges=[GraphEdge.from_dict(e) for e in data.get("edges", [])],
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


# ============================================================================
# Memory Feedback Loop (INT-1)
# ============================================================================


class FeedbackSignal(str, Enum):
    """Feedback signal for memory active learning (INT-1).

    - ``upvote``: Boost importance ×1.15, capped at 1.0.
    - ``downvote``: Penalise importance ×0.85, floor 0.0.
    - ``flag``: Mark as irrelevant — sets ``decay_flag=true``, no immediate importance change.
    - ``positive``: Backward-compatible alias for ``upvote``.
    - ``negative``: Backward-compatible alias for ``downvote``.
    """

    UPVOTE = "upvote"
    DOWNVOTE = "downvote"
    FLAG = "flag"
    POSITIVE = "positive"
    NEGATIVE = "negative"


@dataclass
class FeedbackHistoryEntry:
    """A single recorded feedback event stored in memory metadata (INT-1)."""

    signal: FeedbackSignal
    """Feedback signal that was applied."""
    timestamp: int
    """Unix timestamp (seconds) when feedback was submitted."""
    old_importance: float
    """Memory importance before this feedback was applied."""
    new_importance: float
    """Memory importance after this feedback was applied."""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FeedbackHistoryEntry":
        return cls(
            signal=FeedbackSignal(data["signal"]),
            timestamp=data["timestamp"],
            old_importance=data["old_importance"],
            new_importance=data["new_importance"],
        )


@dataclass
class FeedbackResponse:
    """Response from feedback and importance endpoints (INT-1).

    Returned by ``POST /v1/memories/:id/feedback`` and
    ``PATCH /v1/memories/:id/importance``.
    """

    memory_id: str
    """ID of the memory that was updated."""
    new_importance: float
    """New importance score after the feedback was applied (0.0–1.0)."""
    signal: FeedbackSignal
    """The feedback signal that was recorded."""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FeedbackResponse":
        return cls(
            memory_id=data["memory_id"],
            new_importance=data["new_importance"],
            signal=FeedbackSignal(data["signal"]),
        )


@dataclass
class FeedbackHistoryResponse:
    """Response from ``GET /v1/memories/:id/feedback`` (INT-1)."""

    memory_id: str
    """ID of the memory."""
    entries: list[FeedbackHistoryEntry]
    """Ordered list of feedback events (oldest first, capped at 100)."""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FeedbackHistoryResponse":
        return cls(
            memory_id=data["memory_id"],
            entries=[FeedbackHistoryEntry.from_dict(e) for e in data.get("entries", [])],
        )


@dataclass
class AgentFeedbackSummary:
    """Response from ``GET /v1/agents/:id/feedback/summary`` (INT-1)."""

    agent_id: str
    upvotes: int
    downvotes: int
    flags: int
    total_feedback: int
    health_score: float
    """Weighted-average importance across all non-expired memories (0.0–1.0)."""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentFeedbackSummary":
        return cls(
            agent_id=data["agent_id"],
            upvotes=data["upvotes"],
            downvotes=data["downvotes"],
            flags=data["flags"],
            total_feedback=data["total_feedback"],
            health_score=data["health_score"],
        )


@dataclass
class FeedbackHealthResponse:
    """Response from ``GET /v1/feedback/health`` (INT-1)."""

    agent_id: str
    health_score: float
    """Mean importance of all non-expired memories (0.0–1.0). Higher = healthier."""
    memory_count: int
    avg_importance: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FeedbackHealthResponse":
        return cls(
            agent_id=data["agent_id"],
            health_score=data["health_score"],
            memory_count=data["memory_count"],
            avg_importance=data["avg_importance"],
        )


# =============================================================================
# T-I-F Reliability Scoring (Phase 3 T-I-F RFC)
# =============================================================================


@dataclass
class TifScore:
    """Truth-Indeterminacy-Falsity reliability score for a memory (T-I-F RFC Phase 3).

    Summarises the aggregate feedback signal on a memory as three proportions that
    sum to 1.0.  Use :meth:`from_feedback_history` to compute from the live
    feedback log, or :meth:`from_metadata` to deserialise a value already stored
    in ``memory.metadata["reliability"]``.
    """

    truth: float
    """Proportion of positive feedback signals (upvote / positive)."""
    indeterminacy: float
    """Proportion of uncertainty signals (flag)."""
    falsity: float
    """Proportion of negative feedback signals (downvote / negative)."""
    feedback_count: int
    """Total number of feedback events used to compute this score."""

    @property
    def classification(self) -> str:
        """Classify the memory reliability from the T-I-F proportions.

        Returns one of:
        - ``"surface_contradiction"`` — falsity ≥ 0.50
        - ``"ask_clarification"``    — indeterminacy ≥ 0.50
        - ``"confident_reuse"``      — truth ≥ 0.70
        - ``"verify_before_use"``    — all other cases
        """
        if self.falsity >= 0.50:
            return "surface_contradiction"
        if self.indeterminacy >= 0.50:
            return "ask_clarification"
        if self.truth >= 0.70:
            return "confident_reuse"
        return "verify_before_use"

    @classmethod
    def from_feedback_history(cls, history: "FeedbackHistoryResponse") -> "TifScore":
        """Compute a :class:`TifScore` from a memory's feedback history.

        Signals are bucketed as:
        - ``upvote`` / ``positive`` → truth
        - ``downvote`` / ``negative`` → falsity
        - ``flag`` → indeterminacy

        When no feedback has been recorded the score is
        ``TifScore(truth=0.0, indeterminacy=1.0, falsity=0.0, feedback_count=0)``
        to reflect maximum uncertainty.  When fewer than 3 signals exist, a
        thin-evidence base indeterminacy term is injected and the triple is
        normalised so ``T + I + F == 1.0``.
        """
        upvotes = 0
        downvotes = 0
        flags = 0
        for entry in history.entries:
            sig = (
                entry.signal.value
                if isinstance(entry.signal, FeedbackSignal)
                else str(entry.signal)
            )
            if sig in ("upvote", "positive"):
                upvotes += 1
            elif sig in ("downvote", "negative"):
                downvotes += 1
            elif sig == "flag":
                flags += 1
        total = upvotes + downvotes + flags
        if total == 0:
            return cls(truth=0.0, indeterminacy=1.0, falsity=0.0, feedback_count=0)
        base_indeterminacy = (3 - total) * 0.25 if total < 3 else 0.0
        truth = upvotes / total
        falsity = downvotes / total
        indeterminacy = flags / total + base_indeterminacy
        s = truth + falsity + indeterminacy
        return cls(
            truth=truth / s,
            indeterminacy=indeterminacy / s,
            falsity=falsity / s,
            feedback_count=total,
        )

    @classmethod
    def from_metadata(cls, data: dict[str, Any]) -> "TifScore":
        """Deserialise a :class:`TifScore` from a ``metadata.reliability`` dict.

        Expected keys: ``truth``, ``indeterminacy``, ``falsity``, ``feedback_count``.
        """
        return cls(
            truth=float(data["truth"]),
            indeterminacy=float(data["indeterminacy"]),
            falsity=float(data["falsity"]),
            feedback_count=int(data.get("feedback_count", 0)),
        )


# =============================================================================
# Namespace API Keys (SEC-1)
# =============================================================================


@dataclass
class NamespaceKeyInfo:
    """Namespace-scoped API key metadata (no secret) — SEC-1.

    Returned by :meth:`~dakera.DakeraClient.list_namespace_keys` and
    embedded in :class:`ListNamespaceKeysResponse`.
    """

    key_id: str
    name: str
    namespace: str
    created_at: int
    active: bool
    expires_at: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NamespaceKeyInfo":
        return cls(
            key_id=data["key_id"],
            name=data["name"],
            namespace=data["namespace"],
            created_at=data["created_at"],
            active=data.get("active", True),
            expires_at=data.get("expires_at"),
        )


@dataclass
class CreateNamespaceKeyResponse:
    """Response from ``POST /v1/namespaces/:namespace/keys`` (SEC-1).

    The ``key`` field contains the raw API key and is **shown only once**.
    Store it securely; it cannot be retrieved again.
    """

    key_id: str
    key: str
    name: str
    namespace: str
    created_at: int
    warning: str
    expires_at: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CreateNamespaceKeyResponse":
        return cls(
            key_id=data["key_id"],
            key=data["key"],
            name=data["name"],
            namespace=data["namespace"],
            created_at=data["created_at"],
            warning=data.get("warning", "Save this key — it will not be shown again."),
            expires_at=data.get("expires_at"),
        )


@dataclass
class ListNamespaceKeysResponse:
    """Response from ``GET /v1/namespaces/:namespace/keys`` (SEC-1)."""

    namespace: str
    keys: list[NamespaceKeyInfo]
    total: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ListNamespaceKeysResponse":
        return cls(
            namespace=data["namespace"],
            keys=[NamespaceKeyInfo.from_dict(k) for k in data.get("keys", [])],
            total=data.get("total", 0),
        )


@dataclass
class NamespaceKeyUsageResponse:
    """Response from ``GET /v1/namespaces/:namespace/keys/:key_id/usage`` (SEC-1)."""

    key_id: str
    namespace: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    bytes_transferred: int
    avg_latency_ms: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NamespaceKeyUsageResponse":
        return cls(
            key_id=data["key_id"],
            namespace=data["namespace"],
            total_requests=data.get("total_requests", 0),
            successful_requests=data.get("successful_requests", 0),
            failed_requests=data.get("failed_requests", 0),
            bytes_transferred=data.get("bytes_transferred", 0),
            avg_latency_ms=data.get("avg_latency_ms", 0.0),
        )


# =============================================================================
# CE-6: DBSCAN Adaptive Consolidation
# =============================================================================


@dataclass
class ConsolidationConfig:
    """Algorithm config for DBSCAN adaptive consolidation (CE-6).

    Pass to :meth:`~dakera.DakeraClient.consolidate` via the *config* argument
    to control which clustering algorithm is used.
    """

    algorithm: str | None = None  # "dbscan" | "greedy"
    min_samples: int | None = None
    eps: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        if self.algorithm is not None:
            d["algorithm"] = self.algorithm
        if self.min_samples is not None:
            d["min_samples"] = self.min_samples
        if self.eps is not None:
            d["eps"] = self.eps
        return d


@dataclass
class ConsolidationLogEntry:
    """One step in the consolidation execution log (CE-6)."""

    step: str
    memories_before: int
    memories_after: int
    duration_ms: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConsolidationLogEntry":
        return cls(
            step=data.get("step", ""),
            memories_before=data.get("memories_before", 0),
            memories_after=data.get("memories_after", 0),
            duration_ms=data.get("duration_ms", 0.0),
        )


# =============================================================================
# DX-1: Memory Import / Export
# =============================================================================


@dataclass
class MemoryImportResponse:
    """Response from ``POST /v1/import`` (DX-1)."""

    imported_count: int
    skipped_count: int
    errors: list[str]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryImportResponse":
        return cls(
            imported_count=data.get("imported_count", 0),
            skipped_count=data.get("skipped_count", 0),
            errors=data.get("errors", []),
        )


@dataclass
class MemoryExportResponse:
    """Response from ``GET /v1/export`` (DX-1)."""

    data: list[dict[str, Any]]
    format: str
    count: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryExportResponse":
        raw = data.get("data", [])
        return cls(
            data=raw if isinstance(raw, list) else [],
            format=data.get("format", "jsonl"),
            count=data.get("count", len(raw) if isinstance(raw, list) else 0),
        )


# =============================================================================
# OBS-1: Business-Event Audit Log
# =============================================================================


@dataclass
class AuditEvent:
    """A single business-event entry from the audit log (OBS-1)."""

    id: str
    event_type: str
    agent_id: str | None
    namespace: str | None
    timestamp: int
    details: dict[str, Any]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AuditEvent":
        return cls(
            id=data.get("id", ""),
            event_type=data.get("event_type", ""),
            agent_id=data.get("agent_id"),
            namespace=data.get("namespace"),
            timestamp=data.get("timestamp", 0),
            details=data.get("details", {}),
        )


@dataclass
class AuditListResponse:
    """Response from ``GET /v1/audit`` (OBS-1)."""

    events: list[AuditEvent]
    total: int
    cursor: str | None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AuditListResponse":
        return cls(
            events=[AuditEvent.from_dict(e) for e in data.get("events", [])],
            total=data.get("total", 0),
            cursor=data.get("cursor"),
        )


@dataclass
class AuditExportResponse:
    """Response from ``POST /v1/audit/export`` (OBS-1)."""

    data: str
    format: str
    count: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AuditExportResponse":
        return cls(
            data=data.get("data", ""),
            format=data.get("format", "jsonl"),
            count=data.get("count", 0),
        )


# =============================================================================
# EXT-1: External Extraction Providers
# =============================================================================


@dataclass
class ExtractionResult:
    """Result from ``POST /v1/extract`` (EXT-1)."""

    entities: list[dict[str, Any]]
    provider: str
    model: str | None
    duration_ms: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExtractionResult":
        return cls(
            entities=data.get("entities", []),
            provider=data.get("provider", ""),
            model=data.get("model"),
            duration_ms=data.get("duration_ms", 0.0),
        )


@dataclass
class ExtractionProviderInfo:
    """Metadata for an available extraction provider (EXT-1)."""

    name: str
    available: bool
    models: list[str]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExtractionProviderInfo":
        return cls(
            name=data.get("name", ""),
            available=data.get("available", True),
            models=data.get("models", []),
        )


# =============================================================================
# CE-54: Fulltext Reindex (Admin)
# =============================================================================


@dataclass
class FulltextReindexNamespaceResult:
    """Per-namespace result from ``POST /admin/fulltext/reindex`` (CE-54)."""

    namespace: str
    """Namespace that was scanned."""
    vectors_scanned: int
    """Total vectors examined."""
    newly_indexed: int
    """Memories newly added to the BM25 index."""
    already_indexed: int
    """Memories that were already in the BM25 index."""
    parse_failures: int
    """Memories that could not be parsed (malformed records)."""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FulltextReindexNamespaceResult":
        return cls(
            namespace=data["namespace"],
            vectors_scanned=data.get("vectors_scanned", 0),
            newly_indexed=data.get("newly_indexed", 0),
            already_indexed=data.get("already_indexed", 0),
            parse_failures=data.get("parse_failures", 0),
        )


@dataclass
class FulltextReindexResponse:
    """Response from ``POST /admin/fulltext/reindex`` (CE-54).

    Returned by :meth:`~dakera.DakeraClient.admin_fulltext_reindex`.
    """

    namespaces_processed: int
    """Number of namespaces scanned."""
    total_indexed: int
    """Total memories newly added to BM25 across all namespaces."""
    total_skipped: int
    """Total memories already in the BM25 index (skipped)."""
    details: list[FulltextReindexNamespaceResult]
    """Per-namespace breakdown."""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FulltextReindexResponse":
        return cls(
            namespaces_processed=data.get("namespaces_processed", 0),
            total_indexed=data.get("total_indexed", 0),
            total_skipped=data.get("total_skipped", 0),
            details=[
                FulltextReindexNamespaceResult.from_dict(d) for d in data.get("details", [])
            ],
        )


# =============================================================================
# SEC-3: AES-256-GCM Encryption Key Rotation
# =============================================================================


@dataclass
class RotateEncryptionKeyRequest:
    """Request body for ``POST /v1/admin/encryption/rotate-key`` (SEC-3).

    Args:
        new_key: New passphrase or 64-char hex key to rotate to.
        namespace: If set, rotate only memories in this namespace.
            Omit to rotate all namespaces.
    """

    new_key: str
    namespace: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"new_key": self.new_key}
        if self.namespace is not None:
            d["namespace"] = self.namespace
        return d


@dataclass
class RotateEncryptionKeyResponse:
    """Response from ``POST /v1/admin/encryption/rotate-key`` (SEC-3)."""

    rotated: int
    skipped: int
    namespaces: list[str]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RotateEncryptionKeyResponse":
        return cls(
            rotated=data.get("rotated", 0),
            skipped=data.get("skipped", 0),
            namespaces=data.get("namespaces", []),
        )


# ==============================================================================
# ODE-2: GLiNER Entity Extraction (dakera-ode sidecar)
# ==============================================================================


@dataclass
class OdeEntity:
    """A single entity extracted by the GLiNER model (ODE-2)."""

    text: str
    """Span text as it appears in the input."""
    label: str
    """Entity type label (e.g. ``"person"``, ``"organization"``)."""
    start: int
    """Start character offset (inclusive) within the input text."""
    end: int
    """End character offset (exclusive) within the input text."""
    score: float
    """Confidence score in the range [0, 1]."""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OdeEntity":
        return cls(
            text=data["text"],
            label=data["label"],
            start=int(data["start"]),
            end=int(data["end"]),
            score=float(data["score"]),
        )


@dataclass
class ExtractEntitiesResponse:
    """Response from ``POST /ode/extract`` on the ODE sidecar (ODE-2)."""

    entities: list[OdeEntity]
    """Extracted entities ordered by their start offset."""
    model: str
    """GLiNER model variant used for extraction."""
    processing_time_ms: int
    """Wall-clock time taken by the ODE sidecar in milliseconds."""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExtractEntitiesResponse":
        return cls(
            entities=[OdeEntity.from_dict(e) for e in data.get("entities", [])],
            model=data.get("model", ""),
            processing_time_ms=int(data.get("processing_time_ms", 0)),
        )


# ==============================================================================
# COG-1: Cognitive Memory Lifecycle — per-namespace memory policy
# ==============================================================================


@dataclass
class MemoryPolicy:
    """Per-namespace memory lifecycle policy (COG-1 / COG-3).

    Controls type-specific TTLs, decay curves, spaced repetition behaviour, and
    proactive background consolidation (deduplication).
    All fields have sensible defaults; only override what you need.

    Returned and accepted by :meth:`DakeraClient.get_memory_policy` /
    :meth:`DakeraClient.set_memory_policy` (and their async equivalents).

    .. note::
        ``consolidated_count`` is read-only — the server manages this field.
        Any value you set locally is sent to the API but silently ignored.
    """

    # Differential TTLs -------------------------------------------------------
    working_ttl_seconds: int | None = 14_400
    """Default TTL for ``working`` memories in seconds (default: 4 h)."""
    episodic_ttl_seconds: int | None = 2_592_000
    """Default TTL for ``episodic`` memories in seconds (default: 30 d)."""
    semantic_ttl_seconds: int | None = 31_536_000
    """Default TTL for ``semantic`` memories in seconds (default: 365 d)."""
    procedural_ttl_seconds: int | None = 63_072_000
    """Default TTL for ``procedural`` memories in seconds (default: 730 d)."""

    # Decay curves ------------------------------------------------------------
    working_decay: str = "exponential"
    """Decay strategy for ``working`` memories (default: ``"exponential"``)."""
    episodic_decay: str = "power_law"
    """Decay strategy for ``episodic`` memories (default: ``"power_law"``)."""
    semantic_decay: str = "logarithmic"
    """Decay strategy for ``semantic`` memories (default: ``"logarithmic"``)."""
    procedural_decay: str = "flat"
    """Decay strategy for ``procedural`` memories (default: ``"flat"`` — no decay)."""

    # Spaced repetition -------------------------------------------------------
    spaced_repetition_factor: float = 1.0
    """TTL extension multiplier per recall hit (default: 1.0; set to 0.0 to disable)."""
    spaced_repetition_base_interval_seconds: int = 86_400
    """Base interval in seconds for spaced repetition TTL extension (default: 1 d)."""

    # Proactive consolidation (COG-3) -----------------------------------------
    consolidation_enabled: bool = False
    """Enable background DBSCAN deduplication for this namespace (default: ``False``)."""
    consolidation_threshold: float = 0.92
    """DBSCAN epsilon — cosine-similarity threshold to treat memories as duplicates
    (default: ``0.92``; higher = only merge very close neighbours)."""
    consolidation_interval_hours: int = 24
    """How often (in hours) the background consolidation job runs (default: ``24``)."""
    consolidated_count: int = 0
    """Read-only. Lifetime count of memories merged by the consolidation engine.
    This field is server-managed; any value sent to :meth:`set_memory_policy` is
    silently ignored."""

    # Per-namespace rate limiting (SEC-5) -------------------------------------
    rate_limit_enabled: bool = False
    """Enable per-namespace store/recall rate limiting (default: ``False``)."""
    rate_limit_stores_per_minute: int | None = None
    """Max store operations per minute for this namespace. ``None`` = unlimited (default)."""
    rate_limit_recalls_per_minute: int | None = None
    """Max recall operations per minute for this namespace. ``None`` = unlimited (default)."""

    # Store-time deduplication (CE-10) ----------------------------------------
    dedup_on_store: bool = False
    """Deduplicate against existing memories at store time (CE-10, default: ``False``).

    When ``True`` the server computes a similarity check before persisting a
    new memory and drops it if a near-duplicate already exists (threshold
    controlled by ``dedup_threshold``).
    """
    dedup_threshold: float = 0.92
    """Similarity threshold for store-time deduplication (default: ``0.92``).

    Memories with cosine similarity ≥ this value are considered duplicates
    and the incoming memory is dropped.  Only active when ``dedup_on_store``
    is ``True``.
    """

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-serialisable dict, omitting ``None`` TTL fields."""
        d: dict[str, Any] = {
            "working_decay": self.working_decay,
            "episodic_decay": self.episodic_decay,
            "semantic_decay": self.semantic_decay,
            "procedural_decay": self.procedural_decay,
            "spaced_repetition_factor": self.spaced_repetition_factor,
            "spaced_repetition_base_interval_seconds": self.spaced_repetition_base_interval_seconds,
            "consolidation_enabled": self.consolidation_enabled,
            "consolidation_threshold": self.consolidation_threshold,
            "consolidation_interval_hours": self.consolidation_interval_hours,
            "rate_limit_enabled": self.rate_limit_enabled,
            "dedup_on_store": self.dedup_on_store,
            "dedup_threshold": self.dedup_threshold,
        }
        if self.working_ttl_seconds is not None:
            d["working_ttl_seconds"] = self.working_ttl_seconds
        if self.episodic_ttl_seconds is not None:
            d["episodic_ttl_seconds"] = self.episodic_ttl_seconds
        if self.semantic_ttl_seconds is not None:
            d["semantic_ttl_seconds"] = self.semantic_ttl_seconds
        if self.procedural_ttl_seconds is not None:
            d["procedural_ttl_seconds"] = self.procedural_ttl_seconds
        if self.rate_limit_stores_per_minute is not None:
            d["rate_limit_stores_per_minute"] = self.rate_limit_stores_per_minute
        if self.rate_limit_recalls_per_minute is not None:
            d["rate_limit_recalls_per_minute"] = self.rate_limit_recalls_per_minute
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryPolicy":
        return cls(
            working_ttl_seconds=data.get("working_ttl_seconds"),
            episodic_ttl_seconds=data.get("episodic_ttl_seconds"),
            semantic_ttl_seconds=data.get("semantic_ttl_seconds"),
            procedural_ttl_seconds=data.get("procedural_ttl_seconds"),
            working_decay=data.get("working_decay", "exponential"),
            episodic_decay=data.get("episodic_decay", "power_law"),
            semantic_decay=data.get("semantic_decay", "logarithmic"),
            procedural_decay=data.get("procedural_decay", "flat"),
            spaced_repetition_factor=float(data.get("spaced_repetition_factor", 1.0)),
            spaced_repetition_base_interval_seconds=int(
                data.get("spaced_repetition_base_interval_seconds", 86_400)
            ),
            consolidation_enabled=bool(data.get("consolidation_enabled", False)),
            consolidation_threshold=float(data.get("consolidation_threshold", 0.92)),
            consolidation_interval_hours=int(data.get("consolidation_interval_hours", 24)),
            consolidated_count=int(data.get("consolidated_count", 0)),
            rate_limit_enabled=bool(data.get("rate_limit_enabled", False)),
            rate_limit_stores_per_minute=data.get("rate_limit_stores_per_minute"),
            rate_limit_recalls_per_minute=data.get("rate_limit_recalls_per_minute"),
            dedup_on_store=bool(data.get("dedup_on_store", False)),
            dedup_threshold=float(data.get("dedup_threshold", 0.92)),
        )


# ============================================================================
# Product KPIs (OBS-2)
# ============================================================================


@dataclass
class KpiSnapshot:
    """Point-in-time product KPI snapshot returned by ``GET /v1/kpis`` (OBS-2).

    All latency values are in milliseconds; rate/percentage values are in the
    range ``0.0``–``100.0``. Integer counts are unsigned.

    Requires Admin scope.
    """

    recall_latency_p50_ms: float
    """Median recall latency across all namespaces over the last minute."""
    recall_latency_p99_ms: float
    """99th-percentile recall latency across all namespaces over the last minute."""
    store_latency_p50_ms: float
    """Median store latency across all namespaces over the last minute."""
    api_error_rate_5xx_pct: float
    """5xx error rate as a percentage of total API requests over the last minute."""
    active_agents_count: int
    """Distinct agent identifiers that stored or recalled a memory in the last 24 hours."""
    session_count_week: int
    """Total sessions created in the rolling 7-day window."""
    cross_agent_network_node_count: int
    """Current number of nodes in the cross-agent knowledge graph."""
    memory_retention_7d_pct: float
    """Percentage of memories created 7 days ago that are still active (not decayed or deleted)."""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KpiSnapshot":
        """Construct from API response dict."""
        return cls(
            recall_latency_p50_ms=float(data["recall_latency_p50_ms"]),
            recall_latency_p99_ms=float(data["recall_latency_p99_ms"]),
            store_latency_p50_ms=float(data["store_latency_p50_ms"]),
            api_error_rate_5xx_pct=float(data["api_error_rate_5xx_pct"]),
            active_agents_count=int(data["active_agents_count"]),
            session_count_week=int(data["session_count_week"]),
            cross_agent_network_node_count=int(data["cross_agent_network_node_count"]),
            memory_retention_7d_pct=float(data["memory_retention_7d_pct"]),
        )


# ============================================================================
# Phase 3 — Engine Parity
# ============================================================================


@dataclass
class FullTextIndexStats:
    """Stats for a namespace's full-text index."""

    document_count: int
    unique_terms: int
    avg_doc_length: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FullTextIndexStats":
        """Construct from API response dict."""
        return cls(
            document_count=int(data["document_count"]),
            unique_terms=int(data["unique_terms"]),
            avg_doc_length=float(data["avg_doc_length"]),
        )


@dataclass
class TtlNamespaceStats:
    """TTL stats for a single namespace."""

    namespace: str
    vectors_with_ttl: int
    expiring_within_hour: int
    expiring_within_day: int
    expired_pending_cleanup: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TtlNamespaceStats":
        """Construct from API response dict."""
        return cls(
            namespace=str(data["namespace"]),
            vectors_with_ttl=int(data["vectors_with_ttl"]),
            expiring_within_hour=int(data["expiring_within_hour"]),
            expiring_within_day=int(data["expiring_within_day"]),
            expired_pending_cleanup=int(data["expired_pending_cleanup"]),
        )


@dataclass
class TtlStatsResponse:
    """Aggregated TTL stats across all namespaces."""

    namespaces: list[TtlNamespaceStats]
    total_with_ttl: int
    total_expired: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TtlStatsResponse":
        """Construct from API response dict."""
        return cls(
            namespaces=[TtlNamespaceStats.from_dict(n) for n in data["namespaces"]],
            total_with_ttl=int(data["total_with_ttl"]),
            total_expired=int(data["total_expired"]),
        )


@dataclass
class TtlCleanupResponse:
    """Response from POST /admin/ttl/cleanup."""

    success: bool
    vectors_removed: int
    namespaces_cleaned: list[str]
    message: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TtlCleanupResponse":
        """Construct from API response dict."""
        return cls(
            success=bool(data["success"]),
            vectors_removed=int(data["vectors_removed"]),
            namespaces_cleaned=list(data.get("namespaces_cleaned", [])),
            message=str(data["message"]),
        )


@dataclass
class RouteMatch:
    """A single route match from semantic routing."""

    namespace: str
    similarity: float
    description: str | None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RouteMatch":
        """Construct from API response dict."""
        return cls(
            namespace=str(data["namespace"]),
            similarity=float(data["similarity"]),
            description=data.get("description"),
        )


@dataclass
class RouteResponse:
    """Response from the query routing endpoint."""

    routes: list[RouteMatch]
    model: str
    embedding_time_ms: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RouteResponse":
        """Construct from API response dict."""
        return cls(
            routes=[RouteMatch.from_dict(r) for r in data["routes"]],
            model=str(data["model"]),
            embedding_time_ms=int(data["embedding_time_ms"]),
        )


@dataclass
class ImportJobStatus:
    """Status of an import job."""

    job_id: str
    status: str
    format: str
    total: int
    imported: int
    skipped: int
    errors: list[str]
    started_at: int
    finished_at: int | None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ImportJobStatus":
        """Construct from API response dict."""
        return cls(
            job_id=str(data["job_id"]),
            status=str(data["status"]),
            format=str(data["format"]),
            total=int(data["total"]),
            imported=int(data["imported"]),
            skipped=int(data["skipped"]),
            errors=list(data["errors"]),
            started_at=int(data["started_at"]),
            finished_at=int(data["finished_at"]) if data.get("finished_at") is not None else None,
        )


@dataclass
class TierInfo:
    """Information about a single storage tier."""

    name: str
    tier_type: str
    technology: str
    description: str
    target_latency: str
    capacity: str | None
    status: str
    current_count: int
    hit_count: int
    hit_rate: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TierInfo":
        """Construct from API response dict."""
        return cls(
            name=str(data["name"]),
            tier_type=str(data["tier_type"]),
            technology=str(data["technology"]),
            description=str(data["description"]),
            target_latency=str(data["target_latency"]),
            capacity=data.get("capacity"),
            status=str(data["status"]),
            current_count=int(data["current_count"]),
            hit_count=int(data["hit_count"]),
            hit_rate=float(data["hit_rate"]),
        )


@dataclass
class TierConfig:
    """Storage tier configuration."""

    hot_tier_capacity: int
    hot_to_warm_threshold_secs: int
    warm_to_cold_threshold_secs: int
    auto_tier_enabled: bool
    tier_check_interval_secs: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TierConfig":
        """Construct from API response dict."""
        return cls(
            hot_tier_capacity=int(data["hot_tier_capacity"]),
            hot_to_warm_threshold_secs=int(data["hot_to_warm_threshold_secs"]),
            warm_to_cold_threshold_secs=int(data["warm_to_cold_threshold_secs"]),
            auto_tier_enabled=bool(data["auto_tier_enabled"]),
            tier_check_interval_secs=int(data["tier_check_interval_secs"]),
        )


@dataclass
class TierActivity:
    """Storage tier activity metrics."""

    promotions: int
    demotions: int
    cache_hit_rate: float
    storage_backend: str
    promotions_to_hot: int
    demotions_to_warm: int
    demotions_to_cold: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TierActivity":
        """Construct from API response dict."""
        return cls(
            promotions=int(data["promotions"]),
            demotions=int(data["demotions"]),
            cache_hit_rate=float(data["cache_hit_rate"]),
            storage_backend=str(data["storage_backend"]),
            promotions_to_hot=int(data["promotions_to_hot"]),
            demotions_to_warm=int(data["demotions_to_warm"]),
            demotions_to_cold=int(data["demotions_to_cold"]),
        )


@dataclass
class StorageTierOverview:
    """Full storage tier overview."""

    tiers_enabled: bool
    architecture: list[TierInfo]
    config: TierConfig
    activity: TierActivity

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StorageTierOverview":
        """Construct from API response dict."""
        return cls(
            tiers_enabled=bool(data["tiers_enabled"]),
            architecture=[TierInfo.from_dict(t) for t in data["architecture"]],
            config=TierConfig.from_dict(data["config"]),
            activity=TierActivity.from_dict(data["activity"]),
        )


@dataclass
class MemoryTypeStatsResponse:
    """Memory type distribution stats."""

    total: int
    working: int
    episodic: int
    semantic: int
    procedural: int
    agent_namespaces: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryTypeStatsResponse":
        """Construct from API response dict."""
        return cls(
            total=int(data["total"]),
            working=int(data["working"]),
            episodic=int(data["episodic"]),
            semantic=int(data["semantic"]),
            procedural=int(data["procedural"]),
            agent_namespaces=int(data["agent_namespaces"]),
        )


@dataclass
class NamespaceMigrationResult:
    """Result of migrating a single namespace's dimensions."""

    namespace: str
    original_dimension: int
    vectors_migrated: int
    vectors_skipped: int
    status: str
    error: str | None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NamespaceMigrationResult":
        """Construct from API response dict."""
        return cls(
            namespace=str(data["namespace"]),
            original_dimension=int(data["original_dimension"]),
            vectors_migrated=int(data["vectors_migrated"]),
            vectors_skipped=int(data["vectors_skipped"]),
            status=str(data["status"]),
            error=data.get("error"),
        )


@dataclass
class MigrateDimensionsResponse:
    """Response from the dimension migration endpoint."""

    migrated: int
    failed: int
    already_current: int
    results: list[NamespaceMigrationResult]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MigrateDimensionsResponse":
        """Construct from API response dict."""
        return cls(
            migrated=int(data["migrated"]),
            failed=int(data["failed"]),
            already_current=int(data["already_current"]),
            results=[NamespaceMigrationResult.from_dict(r) for r in data["results"]],
        )


@dataclass
class DrainReembedResponse:
    """Response from ``POST /admin/reembed/drain`` (v0.11.82+).

    Returned once the synchronous re-embedding drain completes (or times out).
    A ``remaining`` of 0 means all static vectors have been upgraded to full
    ONNX quality — suitable as a pre-bench steady-state gate.
    """

    processed: int
    remaining: int
    elapsed_ms: int
    cycles: int
    timed_out: bool

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DrainReembedResponse":
        """Construct from API response dict."""
        return cls(
            processed=int(data["processed"]),
            remaining=int(data["remaining"]),
            elapsed_ms=int(data["elapsed_ms"]),
            cycles=int(data["cycles"]),
            timed_out=bool(data["timed_out"]),
        )


@dataclass
class StaticCountResponse:
    """Response from ``GET /admin/reembed/static-count`` (v0.11.91+).

    Returns the count of static vectors pending re-embedding. Operators
    can poll this during a drain to monitor progress. A ``static_count``
    of 0 means steady state — no vectors awaiting ONNX upgrade.
    """

    static_count: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StaticCountResponse":
        """Construct from API response dict."""
        return cls(static_count=int(data["static_count"]))
