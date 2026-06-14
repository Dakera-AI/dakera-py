"""TealTiger governance middleware integration for Dakera.

Provides persistent, decay-weighted storage for TealTiger governance artefacts
using Dakera's memory API.  Three classes are exported:

- :class:`DakeraCostStorage` — TealTiger ``CostStorage`` ABC backend
- :class:`DakeraDecisionStore` — audit receipt storage with KG-linked retrieval
- :class:`DakeraDelegationHelper` — delegation chain management via memory KG

Usage::

    pip install dakera[tealtiger] tealtiger

    from dakera import DakeraClient
    from dakera.integrations.tealtiger import DakeraCostStorage, DakeraDecisionStore

    client = DakeraClient("http://localhost:3000", api_key="dk-mykey")
    storage = DakeraCostStorage(client)

    # Drop into TealTiger middleware
    from tealtiger import TealTigerMiddleware
    middleware = TealTigerMiddleware(cost_storage=storage)
"""

from __future__ import annotations

import datetime
import json
from typing import Any

from dakera.client import DakeraClient
from dakera.models import BatchForgetRequest, BatchMemoryFilter, BatchRecallRequest

__all__ = [
    "DakeraCostStorage",
    "DakeraDecisionStore",
    "DakeraDelegationHelper",
]

# ---------------------------------------------------------------------------
# Optional TealTiger import — not required at import time
# ---------------------------------------------------------------------------
_HAS_TEALTIGER = False
try:
    import tealtiger  # noqa: F401

    _HAS_TEALTIGER = True
except ImportError:  # pragma: no cover
    pass

# ---------------------------------------------------------------------------
# Importance weights by decision severity
# ---------------------------------------------------------------------------
_DECISION_IMPORTANCE: dict[str, float] = {
    "DENY": 0.95,
    "REQUIRE_APPROVAL": 0.90,
    "ALLOW": 0.80,
    "MODIFY": 0.80,
}
_COST_IMPORTANCE = 0.7
_GOVERNANCE_TAGS = ["governance", "cost"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_unix(dt: datetime.datetime | str | None) -> int | None:
    """Convert a datetime or ISO-8601 string to a Unix timestamp (seconds)."""
    if dt is None:
        return None
    if isinstance(dt, str):
        dt = datetime.datetime.fromisoformat(dt.replace("Z", "+00:00"))
    return int(dt.timestamp())


def _provider_str(provider: Any) -> str:
    return provider.value if hasattr(provider, "value") else str(provider)


def _model_dump(obj: Any) -> str:
    if hasattr(obj, "model_dump_json"):
        return obj.model_dump_json()
    return json.dumps(obj)


# ---------------------------------------------------------------------------
# DakeraCostStorage
# ---------------------------------------------------------------------------


class DakeraCostStorage:
    """TealTiger ``CostStorage`` backed by Dakera persistent agent memory.

    All cost records are stored in a single Dakera namespace (``dakera_agent_id``,
    default ``"governance"``) and tagged with provider, model, agent, and cost_id
    for efficient lookups without full-table scans.

    When TealTiger is installed this class registers as a virtual subclass of
    ``tealtiger.cost.storage.CostStorage`` so ``isinstance()`` checks pass
    without requiring ABC inheritance.  When TealTiger is absent, the class still
    works for storage and retrieval but returns raw dicts from ``get()`` instead
    of typed ``CostRecord`` objects.

    Args:
        client: An initialised :class:`~dakera.DakeraClient`.
        dakera_agent_id: Dakera namespace to store all cost records in.
            Must be consistent across calls.  Defaults to ``"governance"``.

    Example::

        client = DakeraClient("http://localhost:3000", api_key="dk-key")
        storage = DakeraCostStorage(client)

        from tealtiger import TealTigerMiddleware
        middleware = TealTigerMiddleware(cost_storage=storage)
    """

    def __init__(
        self,
        client: DakeraClient,
        dakera_agent_id: str = "governance",
    ) -> None:
        self._client = client
        self._agent_id = dakera_agent_id

    # ------------------------------------------------------------------
    # CostStorage ABC — all 8 methods
    # ------------------------------------------------------------------

    def store(self, record: Any) -> None:
        """Persist a ``CostRecord`` as a tagged Dakera memory."""
        tags = [
            *_GOVERNANCE_TAGS,
            f"model:{record.model}",
            f"provider:{_provider_str(record.provider)}",
            f"cost_id:{record.id}",
            f"request_id:{record.request_id}",
            f"agent:{record.agent_id}",
        ]
        self._client.store_memory(
            agent_id=self._agent_id,
            content=_model_dump(record),
            importance=_COST_IMPORTANCE,
            tags=tags,
        )

    def get(self, id: str) -> Any | None:
        """Retrieve a single ``CostRecord`` by its ID."""
        resp = self._client.batch_recall(
            BatchRecallRequest(
                agent_id=self._agent_id,
                filter=BatchMemoryFilter(tags=["cost", f"cost_id:{id}"]),
                limit=1,
            )
        )
        if not resp.memories:
            return None
        return self._deserialize(resp.memories[0].content)

    def get_by_request_id(self, request_id: str) -> list[Any]:
        """Retrieve all ``CostRecord``\\s for a given request ID."""
        resp = self._client.batch_recall(
            BatchRecallRequest(
                agent_id=self._agent_id,
                filter=BatchMemoryFilter(tags=["cost", f"request_id:{request_id}"]),
                limit=1000,
            )
        )
        return [r for r in (self._deserialize(m.content) for m in resp.memories) if r is not None]

    def get_by_agent_id(
        self,
        agent_id: str,
        start_date: datetime.datetime | str | None = None,
        end_date: datetime.datetime | str | None = None,
    ) -> list[Any]:
        """Retrieve all ``CostRecord``\\s for a specific agent, with optional date bounds."""
        resp = self._client.batch_recall(
            BatchRecallRequest(
                agent_id=self._agent_id,
                filter=BatchMemoryFilter(
                    tags=["cost", f"agent:{agent_id}"],
                    created_after=_to_unix(start_date),
                    created_before=_to_unix(end_date),
                ),
                limit=1000,
            )
        )
        return [r for r in (self._deserialize(m.content) for m in resp.memories) if r is not None]

    def get_by_date_range(
        self,
        start_date: datetime.datetime | str,
        end_date: datetime.datetime | str,
    ) -> list[Any]:
        """Retrieve all ``CostRecord``\\s within a date range."""
        resp = self._client.batch_recall(
            BatchRecallRequest(
                agent_id=self._agent_id,
                filter=BatchMemoryFilter(
                    tags=_GOVERNANCE_TAGS,
                    created_after=_to_unix(start_date),
                    created_before=_to_unix(end_date),
                ),
                limit=1000,
            )
        )
        return [r for r in (self._deserialize(m.content) for m in resp.memories) if r is not None]

    def get_summary(
        self,
        start_date: datetime.datetime | str,
        end_date: datetime.datetime | str,
        agent_id: str | None = None,
    ) -> Any:
        """Return aggregated cost statistics for a date range.

        Performs client-side aggregation matching ``InMemoryCostStorage`` logic.
        Returns a TealTiger ``CostSummary`` when the package is installed, or a
        plain ``dict`` otherwise.
        """
        tags = ["cost", f"agent:{agent_id}"] if agent_id else _GOVERNANCE_TAGS
        resp = self._client.batch_recall(
            BatchRecallRequest(
                agent_id=self._agent_id,
                filter=BatchMemoryFilter(
                    tags=tags,
                    created_after=_to_unix(start_date),
                    created_before=_to_unix(end_date),
                ),
                limit=1000,
            )
        )
        records = [
            r for r in (self._deserialize(m.content) for m in resp.memories) if r is not None
        ]
        return self._aggregate_summary(records, start_date, end_date)

    def delete_older_than(self, before_date: datetime.datetime | str) -> int:
        """Delete all ``CostRecord``\\s older than *before_date*. Returns deleted count."""
        resp = self._client.batch_forget(
            BatchForgetRequest(
                agent_id=self._agent_id,
                filter=BatchMemoryFilter(
                    tags=_GOVERNANCE_TAGS,
                    created_before=_to_unix(before_date),
                ),
            )
        )
        return resp.deleted_count

    def clear(self) -> None:
        """Delete all ``CostRecord``\\s in this storage."""
        self._client.batch_forget(
            BatchForgetRequest(
                agent_id=self._agent_id,
                filter=BatchMemoryFilter(tags=_GOVERNANCE_TAGS),
            )
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _deserialize(self, content: str) -> Any | None:
        if _HAS_TEALTIGER:
            try:
                from tealtiger.cost.record import CostRecord  # noqa: PLC0415

                return CostRecord.model_validate_json(content)
            except Exception:
                return None
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return None

    def _aggregate_summary(
        self,
        records: list[Any],
        start_date: Any,
        end_date: Any,
    ) -> Any:
        total_cost = 0.0
        total_tokens = 0
        by_model: dict[str, float] = {}
        by_provider: dict[str, float] = {}
        by_agent: dict[str, float] = {}

        for rec in records:
            if hasattr(rec, "actual_cost"):
                cost = rec.actual_cost
                model = rec.model
                provider = _provider_str(rec.provider)
                agent = rec.agent_id
                if hasattr(rec, "actual_tokens") and rec.actual_tokens is not None:
                    tok = rec.actual_tokens
                    total_tokens += (
                        tok.get("total", 0) if isinstance(tok, dict) else getattr(tok, "total", 0)
                    )
            else:
                cost = rec.get("actual_cost", 0.0)
                model = rec.get("model", "unknown")
                provider = rec.get("provider", "unknown")
                agent = rec.get("agent_id", "unknown")

            total_cost += cost
            by_model[model] = by_model.get(model, 0.0) + cost
            by_provider[provider] = by_provider.get(provider, 0.0) + cost
            by_agent[agent] = by_agent.get(agent, 0.0) + cost

        total = len(records)
        avg = total_cost / total if total > 0 else 0.0
        period = f"{start_date}/{end_date}"

        if _HAS_TEALTIGER:
            try:
                from tealtiger.cost.record import CostSummary  # noqa: PLC0415

                return CostSummary(
                    total_cost=total_cost,
                    total_requests=total,
                    average_cost_per_request=avg,
                    by_model=by_model,
                    by_provider=by_provider,
                    by_agent=by_agent,
                    period=period,
                    total_tokens=total_tokens,
                )
            except Exception:
                pass

        return {
            "total_cost": total_cost,
            "total_requests": total,
            "average_cost_per_request": avg,
            "by_model": by_model,
            "by_provider": by_provider,
            "by_agent": by_agent,
            "period": period,
            "total_tokens": total_tokens,
        }


# Register as a virtual CostStorage subclass so isinstance() checks pass
# when TealTiger is installed, without requiring ABC inheritance at class definition.
try:
    from tealtiger.cost.storage import CostStorage as _TtCostStorage

    _TtCostStorage.register(DakeraCostStorage)
except ImportError:  # pragma: no cover
    pass


# ---------------------------------------------------------------------------
# DakeraDecisionStore
# ---------------------------------------------------------------------------


class DakeraDecisionStore:
    """Stores and retrieves TealTiger governance decision receipts in Dakera memory.

    Decision receipts are stored as episodic memories with importance weighted
    by decision severity:

    - ``DENY`` → 0.95 (highest importance, longest retention)
    - ``REQUIRE_APPROVAL`` → 0.90
    - ``ALLOW`` / ``MODIFY`` → 0.80

    This ensures deny-decisions survive memory compaction longer, maintaining
    a robust audit trail for compliance.

    Args:
        client: An initialised :class:`~dakera.DakeraClient`.

    Example::

        store = DakeraDecisionStore(client)
        mem_id = store.store_receipt("my-agent", receipt)
        found = store.lookup_receipt("my-agent", "dec-abc123")
        duplicate = store.is_terminal("my-agent", "idem-key-xyz")
    """

    def __init__(self, client: DakeraClient) -> None:
        self._client = client

    def store_receipt(self, agent_id: str, receipt: Any) -> str:
        """Persist a TealTiger governance receipt.

        Args:
            agent_id: Dakera namespace (agent) to store the receipt under.
            receipt: A TealTiger ``GovernanceReceipt`` or compatible object.

        Returns:
            The Dakera memory ID of the stored receipt.
        """
        decision_raw = (
            receipt.decision.value if hasattr(receipt.decision, "value") else str(receipt.decision)
        )
        importance = _DECISION_IMPORTANCE.get(decision_raw.upper(), _DECISION_IMPORTANCE["ALLOW"])
        decision_id = str(getattr(receipt, "decision_id", ""))
        idempotency_key = str(getattr(receipt, "idempotency_key", ""))
        tags = [
            "governance",
            "decision",
            f"decision:{decision_raw.lower()}",
            f"decision_id:{decision_id}",
            f"idempotency:{idempotency_key}",
        ]
        mem = self._client.store_memory(
            agent_id=agent_id,
            content=_model_dump(receipt),
            importance=importance,
            memory_type="episodic",
            tags=tags,
        )
        return str(mem.get("id", ""))

    def lookup_receipt(self, agent_id: str, decision_id: str) -> Any | None:
        """Look up a governance receipt by decision ID.

        Args:
            agent_id: Dakera namespace owning the receipt.
            decision_id: The TealTiger decision ID to search for.

        Returns:
            A ``GovernanceReceipt`` (if TealTiger is installed) or raw dict,
            or ``None`` if not found.
        """
        resp = self._client.batch_recall(
            BatchRecallRequest(
                agent_id=agent_id,
                filter=BatchMemoryFilter(
                    tags=["governance", "decision", f"decision_id:{decision_id}"]
                ),
                limit=1,
            )
        )
        if not resp.memories:
            return None
        content = resp.memories[0].content
        if _HAS_TEALTIGER:
            try:
                from tealtiger.governance.receipt import GovernanceReceipt  # noqa: PLC0415

                return GovernanceReceipt.model_validate_json(content)
            except Exception:
                pass
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return None

    def is_terminal(self, agent_id: str, idempotency_key: str) -> bool:
        """Return ``True`` if a decision for *idempotency_key* is already stored.

        Use this for idempotency checks before evaluating governance rules to
        avoid processing the same request twice.

        Args:
            agent_id: Dakera namespace to search.
            idempotency_key: The idempotency key from the incoming request.
        """
        resp = self._client.batch_recall(
            BatchRecallRequest(
                agent_id=agent_id,
                filter=BatchMemoryFilter(
                    tags=["governance", "decision", f"idempotency:{idempotency_key}"]
                ),
                limit=1,
            )
        )
        return len(resp.memories) > 0


# ---------------------------------------------------------------------------
# DakeraDelegationHelper
# ---------------------------------------------------------------------------


class DakeraDelegationHelper:
    """Manages agent delegation chains using the Dakera memory knowledge graph.

    Creates typed ``delegated_from`` edges between decision receipt memory nodes,
    enabling audit-trail traversal across arbitrarily deep delegation hierarchies.

    Args:
        client: An initialised :class:`~dakera.DakeraClient`.

    Example::

        helper = DakeraDelegationHelper(client)

        # Link child receipt to parent
        helper.link_delegation(child_id=child_mem_id, parent_id=parent_mem_id)

        # Traverse the full chain
        chain = helper.get_delegation_chain("my-agent", root_mem_id, max_depth=5)
        # ["root-mem-id", "parent-mem-id", "grandparent-mem-id"]
    """

    _EDGE_TYPE = "delegated_from"

    def __init__(self, client: DakeraClient) -> None:
        self._client = client

    def link_delegation(self, child_id: str, parent_id: str) -> None:
        """Create a ``delegated_from`` KG edge from *child_id* to *parent_id*.

        Args:
            child_id: Dakera memory ID of the child (delegated) receipt.
            parent_id: Dakera memory ID of the parent (delegating) receipt.
        """
        self._client.memory_link(
            source_id=child_id,
            target_id=parent_id,
            edge_type=self._EDGE_TYPE,
        )

    def get_delegation_chain(
        self,
        agent_id: str,
        decision_id: str,
        max_depth: int = 10,
    ) -> list[str]:
        """Traverse the delegation chain from a root decision receipt.

        Performs a BFS traversal over ``delegated_from`` edges in the memory KG,
        returning an ordered list of memory IDs from root outward.

        Args:
            agent_id: Dakera namespace containing the receipt memories.
            decision_id: Dakera memory ID of the root decision receipt.
            max_depth: Maximum hops to traverse (clamped to 5 by the KG API;
                values above 5 will silently use 5).

        Returns:
            Ordered list of memory IDs starting with *decision_id*.
        """
        result = self._client.knowledge_query(
            agent_id=agent_id,
            root_id=decision_id,
            edge_type=self._EDGE_TYPE,
            max_depth=min(max_depth, 5),
        )
        seen: set[str] = {decision_id}
        chain: list[str] = [decision_id]
        for edge in result.edges:
            for nid in (edge.source_id, edge.target_id):
                if nid not in seen:
                    seen.add(nid)
                    chain.append(nid)
        return chain
