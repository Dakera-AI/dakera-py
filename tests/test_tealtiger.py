"""Tests for TealTiger governance middleware integration.

All tests use mocked DakeraClient — TealTiger does NOT need to be installed.
Tests pass BOTH with and without tealtiger installed.
"""

import json
from unittest.mock import MagicMock

from dakera.integrations.tealtiger import (
    DakeraCostStorage,
    DakeraDecisionStore,
    DakeraDelegationHelper,
)
from dakera.models import (
    BatchForgetResponse,
    BatchRecallResponse,
    EdgeType,
    GraphEdge,
    KgQueryResponse,
    Memory,
)

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

# Complete CostRecord JSON (all required fields per tealtiger v1.3.0 schema).
_COMPLETE_COST_JSON = {
    "id": "cost-1",
    "request_id": "req-1",
    "agent_id": "ag-1",
    "model": "gpt-4o",
    "provider": "openai",
    "actual_tokens": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
    "actual_cost": 0.05,
    "breakdown": {"input_cost": 0.03, "output_cost": 0.02},
    "timestamp": "2026-06-14T12:00:00Z",
}


def _make_client() -> MagicMock:
    """Return a mock DakeraClient."""
    return MagicMock()


def _make_cost_record(
    cost_id: str = "cost-1",
    request_id: str = "req-1",
    agent_id: str = "ag-1",
    model: str = "gpt-4o",
    provider: str = "openai",
    cost: float = 0.05,
) -> MagicMock:
    record = MagicMock()
    record.id = cost_id
    record.request_id = request_id
    record.agent_id = agent_id
    record.model = model
    record.provider.value = provider
    record.actual_cost = cost
    # Include all required CostRecord fields so model_validate_json succeeds
    # when tealtiger IS installed.
    record.actual_tokens = MagicMock(input_tokens=100, output_tokens=50, total_tokens=150)
    record.model_dump_json.return_value = json.dumps(
        {
            "id": cost_id,
            "request_id": request_id,
            "agent_id": agent_id,
            "model": model,
            "provider": provider,
            "actual_tokens": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
            "actual_cost": cost,
            "breakdown": {"input_cost": 0.03, "output_cost": 0.02},
            "timestamp": "2026-06-14T12:00:00Z",
        }
    )
    return record


def _make_batch_recall_response(memories: list[Memory] | None = None) -> BatchRecallResponse:
    mems = memories or []
    return BatchRecallResponse(memories=mems, total=len(mems), filtered=len(mems))


def _make_memory(content: str, memory_id: str = "mem-1") -> Memory:
    return Memory(
        id=memory_id,
        content=content,
        memory_type="episodic",
        importance=0.7,
    )


def _complete_cost_payload(**overrides: object) -> str:
    """Return a complete CostRecord JSON string (passes model_validate_json)."""
    data = dict(_COMPLETE_COST_JSON)
    data.update(overrides)  # type: ignore[arg-type]
    return json.dumps(data)


# ---------------------------------------------------------------------------
# DakeraCostStorage
# ---------------------------------------------------------------------------


class TestDakeraCostStorageStore:
    def test_store_calls_store_memory_with_correct_tags(self) -> None:
        client = _make_client()
        storage = DakeraCostStorage(client, dakera_agent_id="test-ns")
        record = _make_cost_record()

        storage.store(record)

        client.store_memory.assert_called_once()
        call_kwargs = client.store_memory.call_args.kwargs
        assert call_kwargs["agent_id"] == "test-ns"
        assert call_kwargs["importance"] == 0.7
        tags = call_kwargs["tags"]
        assert "governance" in tags
        assert "cost" in tags
        assert "model:gpt-4o" in tags
        assert "provider:openai" in tags
        assert "cost_id:cost-1" in tags
        assert "request_id:req-1" in tags
        assert "agent:ag-1" in tags

    def test_store_content_is_json(self) -> None:
        client = _make_client()
        storage = DakeraCostStorage(client)
        record = _make_cost_record(cost_id="c-99")

        storage.store(record)

        content = client.store_memory.call_args.kwargs["content"]
        parsed = json.loads(content)
        assert parsed["id"] == "c-99"


class TestDakeraCostStorageGet:
    def test_get_returns_none_when_not_found(self) -> None:
        client = _make_client()
        client.batch_recall.return_value = _make_batch_recall_response()
        storage = DakeraCostStorage(client)

        result = storage.get("missing-id")

        assert result is None
        client.batch_recall.assert_called_once()

    def test_get_passes_correct_filter(self) -> None:
        # Use complete JSON so model_validate_json succeeds when tealtiger is installed.
        payload = _complete_cost_payload(id="cost-1", actual_cost=0.05)
        client = _make_client()
        client.batch_recall.return_value = _make_batch_recall_response(
            [_make_memory(payload)]
        )
        storage = DakeraCostStorage(client)

        result = storage.get("cost-1")

        assert result is not None
        req = client.batch_recall.call_args.args[0]
        assert req.filter is not None
        assert "cost_id:cost-1" in (req.filter.tags or [])

    def test_get_deserializes_json_when_tealtiger_absent(self) -> None:
        # Patch _HAS_TEALTIGER to False for this test to verify dict fallback.
        import dakera.integrations.tealtiger as tt_mod

        original = tt_mod._HAS_TEALTIGER
        tt_mod._HAS_TEALTIGER = False
        try:
            payload = json.dumps({"id": "cost-1", "actual_cost": 0.05, "model": "gpt-4o"})
            client = _make_client()
            client.batch_recall.return_value = _make_batch_recall_response(
                [_make_memory(payload)]
            )
            storage = DakeraCostStorage(client)

            result = storage.get("cost-1")

            assert isinstance(result, dict)
            assert result["model"] == "gpt-4o"
        finally:
            tt_mod._HAS_TEALTIGER = original


class TestDakeraCostStorageGetByRequestId:
    def test_returns_empty_list_when_none_found(self) -> None:
        client = _make_client()
        client.batch_recall.return_value = _make_batch_recall_response()
        storage = DakeraCostStorage(client)

        result = storage.get_by_request_id("req-xyz")

        assert result == []

    def test_filter_includes_request_id_tag(self) -> None:
        client = _make_client()
        client.batch_recall.return_value = _make_batch_recall_response()
        storage = DakeraCostStorage(client)

        storage.get_by_request_id("req-abc")

        req = client.batch_recall.call_args.args[0]
        assert "request_id:req-abc" in (req.filter.tags or [])


class TestDakeraCostStorageGetByAgentId:
    def test_filter_includes_agent_tag(self) -> None:
        client = _make_client()
        client.batch_recall.return_value = _make_batch_recall_response()
        storage = DakeraCostStorage(client)

        storage.get_by_agent_id("my-agent")

        req = client.batch_recall.call_args.args[0]
        assert "agent:my-agent" in (req.filter.tags or [])

    def test_date_bounds_forwarded(self) -> None:
        client = _make_client()
        client.batch_recall.return_value = _make_batch_recall_response()
        storage = DakeraCostStorage(client)

        storage.get_by_agent_id(
            "ag-1", start_date="2026-01-01T00:00:00Z", end_date="2026-12-31T23:59:59Z"
        )

        req = client.batch_recall.call_args.args[0]
        assert req.filter.created_after is not None
        assert req.filter.created_before is not None


class TestDakeraCostStorageGetByDateRange:
    def test_filter_uses_governance_tags(self) -> None:
        client = _make_client()
        client.batch_recall.return_value = _make_batch_recall_response()
        storage = DakeraCostStorage(client)

        storage.get_by_date_range("2026-01-01T00:00:00Z", "2026-12-31T23:59:59Z")

        req = client.batch_recall.call_args.args[0]
        assert "governance" in (req.filter.tags or [])
        assert "cost" in (req.filter.tags or [])
        assert req.filter.created_after is not None
        assert req.filter.created_before is not None


class TestDakeraCostStorageGetSummary:
    def test_returns_dict_when_tealtiger_absent(self) -> None:
        import dakera.integrations.tealtiger as tt_mod

        original = tt_mod._HAS_TEALTIGER
        tt_mod._HAS_TEALTIGER = False
        try:
            payload = json.dumps(
                {
                    "actual_cost": 0.10,
                    "model": "gpt-4o",
                    "provider": "openai",
                    "agent_id": "ag-1",
                }
            )
            client = _make_client()
            client.batch_recall.return_value = _make_batch_recall_response(
                [_make_memory(payload), _make_memory(payload, "mem-2")]
            )
            storage = DakeraCostStorage(client)

            result = storage.get_summary("2026-01-01T00:00:00Z", "2026-12-31T23:59:59Z")

            assert isinstance(result, dict)
            assert abs(result["total_cost"] - 0.20) < 1e-9
            assert result["total_requests"] == 2
            assert "gpt-4o" in result["by_model"]
        finally:
            tt_mod._HAS_TEALTIGER = original

    def test_agent_id_filter_applied_when_given(self) -> None:
        client = _make_client()
        client.batch_recall.return_value = _make_batch_recall_response()
        storage = DakeraCostStorage(client)

        storage.get_summary("2026-01-01T00:00:00Z", "2026-12-31T23:59:59Z", agent_id="ag-X")

        req = client.batch_recall.call_args.args[0]
        assert "agent:ag-X" in (req.filter.tags or [])

    def test_period_is_dict(self) -> None:
        import dakera.integrations.tealtiger as tt_mod

        original = tt_mod._HAS_TEALTIGER
        tt_mod._HAS_TEALTIGER = False
        try:
            client = _make_client()
            client.batch_recall.return_value = _make_batch_recall_response()
            storage = DakeraCostStorage(client)

            result = storage.get_summary("2026-01-01T00:00:00Z", "2026-12-31T23:59:59Z")

            assert isinstance(result["period"], dict)
            assert "start" in result["period"]
            assert "end" in result["period"]
        finally:
            tt_mod._HAS_TEALTIGER = original

    def test_total_tokens_is_dict(self) -> None:
        import dakera.integrations.tealtiger as tt_mod

        original = tt_mod._HAS_TEALTIGER
        tt_mod._HAS_TEALTIGER = False
        try:
            client = _make_client()
            client.batch_recall.return_value = _make_batch_recall_response()
            storage = DakeraCostStorage(client)

            result = storage.get_summary("2026-01-01T00:00:00Z", "2026-12-31T23:59:59Z")

            assert isinstance(result["total_tokens"], dict)
            assert "total" in result["total_tokens"]
        finally:
            tt_mod._HAS_TEALTIGER = original


class TestDakeraCostStorageDeleteOlderThan:
    def test_calls_batch_forget_with_created_before(self) -> None:
        client = _make_client()
        client.batch_forget.return_value = BatchForgetResponse(deleted_count=3)
        storage = DakeraCostStorage(client)

        count = storage.delete_older_than("2026-01-01T00:00:00Z")

        assert count == 3
        req = client.batch_forget.call_args.args[0]
        assert req.filter.created_before is not None
        assert "governance" in (req.filter.tags or [])


class TestDakeraCostStorageClear:
    def test_calls_batch_forget_with_governance_tags(self) -> None:
        client = _make_client()
        client.batch_forget.return_value = BatchForgetResponse(deleted_count=10)
        storage = DakeraCostStorage(client)

        storage.clear()

        client.batch_forget.assert_called_once()
        req = client.batch_forget.call_args.args[0]
        assert "governance" in (req.filter.tags or [])
        assert "cost" in (req.filter.tags or [])


# ---------------------------------------------------------------------------
# DakeraDecisionStore
# ---------------------------------------------------------------------------


def _make_decision(
    action: str = "DENY",
    correlation_id: str = "corr-1",
    policy_id: str = "policy-1",
) -> MagicMock:
    """Return a mock tealtiger Decision object."""
    decision = MagicMock()
    decision.action.value = action
    decision.correlation_id = correlation_id
    decision.policy_id = policy_id
    decision.model_dump_json.return_value = json.dumps(
        {
            "action": action,
            "correlation_id": correlation_id,
            "policy_id": policy_id,
            "risk_score": 50,
            "reason": "test",
        }
    )
    return decision


class TestDakeraDecisionStoreStoreReceipt:
    def test_deny_uses_highest_importance(self) -> None:
        client = _make_client()
        client.store_memory.return_value = {"id": "mem-deny-1"}
        ds = DakeraDecisionStore(client)

        ds.store_receipt("ag-1", _make_decision("DENY"))

        kwargs = client.store_memory.call_args.kwargs
        assert kwargs["importance"] == 0.95

    def test_require_approval_importance(self) -> None:
        client = _make_client()
        client.store_memory.return_value = {"id": "mem-ra-1"}
        ds = DakeraDecisionStore(client)

        ds.store_receipt("ag-1", _make_decision("REQUIRE_APPROVAL"))

        assert client.store_memory.call_args.kwargs["importance"] == 0.90

    def test_redact_importance(self) -> None:
        client = _make_client()
        client.store_memory.return_value = {"id": "mem-redact-1"}
        ds = DakeraDecisionStore(client)

        ds.store_receipt("ag-1", _make_decision("REDACT"))

        assert client.store_memory.call_args.kwargs["importance"] == 0.90

    def test_transform_importance(self) -> None:
        client = _make_client()
        client.store_memory.return_value = {"id": "mem-transform-1"}
        ds = DakeraDecisionStore(client)

        ds.store_receipt("ag-1", _make_decision("TRANSFORM"))

        assert client.store_memory.call_args.kwargs["importance"] == 0.85

    def test_degrade_importance(self) -> None:
        client = _make_client()
        client.store_memory.return_value = {"id": "mem-degrade-1"}
        ds = DakeraDecisionStore(client)

        ds.store_receipt("ag-1", _make_decision("DEGRADE"))

        assert client.store_memory.call_args.kwargs["importance"] == 0.85

    def test_allow_importance(self) -> None:
        client = _make_client()
        client.store_memory.return_value = {"id": "mem-allow-1"}
        ds = DakeraDecisionStore(client)

        ds.store_receipt("ag-1", _make_decision("ALLOW"))

        assert client.store_memory.call_args.kwargs["importance"] == 0.80

    def test_tags_include_decision_and_ids(self) -> None:
        client = _make_client()
        client.store_memory.return_value = {"id": "mem-1"}
        ds = DakeraDecisionStore(client)

        ds.store_receipt("ag-1", _make_decision("DENY", "corr-99", "policy-99"))

        tags = client.store_memory.call_args.kwargs["tags"]
        assert "governance" in tags
        assert "decision" in tags
        assert "decision:deny" in tags
        assert "correlation_id:corr-99" in tags
        assert "policy_id:policy-99" in tags

    def test_returns_memory_id(self) -> None:
        client = _make_client()
        client.store_memory.return_value = {"id": "mem-abc"}
        ds = DakeraDecisionStore(client)

        mem_id = ds.store_receipt("ag-1", _make_decision())

        assert mem_id == "mem-abc"


class TestDakeraDecisionStoreLookupReceipt:
    def test_returns_none_when_not_found(self) -> None:
        client = _make_client()
        client.batch_recall.return_value = _make_batch_recall_response()
        ds = DakeraDecisionStore(client)

        result = ds.lookup_receipt("ag-1", "corr-missing")

        assert result is None

    def test_filter_includes_correlation_id(self) -> None:
        payload = json.dumps(
            {"action": "ALLOW", "correlation_id": "corr-42", "policy_id": "p-1"}
        )
        client = _make_client()
        client.batch_recall.return_value = _make_batch_recall_response(
            [_make_memory(payload)]
        )
        ds = DakeraDecisionStore(client)

        result = ds.lookup_receipt("ag-1", "corr-42")

        assert result is not None
        req = client.batch_recall.call_args.args[0]
        assert "correlation_id:corr-42" in (req.filter.tags or [])


class TestDakeraDecisionStoreIsTerminal:
    def test_returns_false_when_no_receipt(self) -> None:
        client = _make_client()
        client.batch_recall.return_value = _make_batch_recall_response()
        ds = DakeraDecisionStore(client)

        assert ds.is_terminal("ag-1", "corr-new") is False

    def test_returns_true_when_receipt_exists(self) -> None:
        payload = json.dumps({"correlation_id": "corr-exists"})
        client = _make_client()
        client.batch_recall.return_value = _make_batch_recall_response(
            [_make_memory(payload)]
        )
        ds = DakeraDecisionStore(client)

        assert ds.is_terminal("ag-1", "corr-exists") is True

    def test_filter_includes_correlation_id_tag(self) -> None:
        client = _make_client()
        client.batch_recall.return_value = _make_batch_recall_response()
        ds = DakeraDecisionStore(client)

        ds.is_terminal("ag-1", "corr-xyz")

        req = client.batch_recall.call_args.args[0]
        assert "correlation_id:corr-xyz" in (req.filter.tags or [])


# ---------------------------------------------------------------------------
# DakeraDelegationHelper
# ---------------------------------------------------------------------------


class TestDakeraDelegationHelperLink:
    def test_link_delegation_calls_memory_link(self) -> None:
        client = _make_client()
        helper = DakeraDelegationHelper(client)

        helper.link_delegation(child_id="child-mem", parent_id="parent-mem")

        client.memory_link.assert_called_once_with(
            source_id="child-mem",
            target_id="parent-mem",
            edge_type="delegated_from",
        )


class TestDakeraDelegationHelperGetChain:
    def _make_kg_response(self, root: str, hops: list[str]) -> KgQueryResponse:
        edges = []
        for i, hop in enumerate(hops):
            src = root if i == 0 else hops[i - 1]
            edges.append(
                GraphEdge(
                    id=f"edge-{i}",
                    source_id=src,
                    target_id=hop,
                    edge_type=EdgeType.LINKED_BY,
                    weight=1.0,
                    created_at=0,
                )
            )
        return KgQueryResponse(
            agent_id="ag-1",
            node_count=len(hops) + 1,
            edge_count=len(edges),
            edges=edges,
        )

    def test_chain_starts_with_decision_id(self) -> None:
        client = _make_client()
        client.knowledge_query.return_value = self._make_kg_response(
            "root", ["parent", "grandparent"]
        )
        helper = DakeraDelegationHelper(client)

        chain = helper.get_delegation_chain("ag-1", "root", max_depth=5)

        assert chain[0] == "root"
        assert "parent" in chain
        assert "grandparent" in chain

    def test_max_depth_clamped_to_5(self) -> None:
        client = _make_client()
        client.knowledge_query.return_value = KgQueryResponse(
            agent_id="ag-1", node_count=0, edge_count=0, edges=[]
        )
        helper = DakeraDelegationHelper(client)

        helper.get_delegation_chain("ag-1", "root", max_depth=20)

        call_kwargs = client.knowledge_query.call_args.kwargs
        assert call_kwargs["max_depth"] <= 5

    def test_no_edges_returns_single_item_chain(self) -> None:
        client = _make_client()
        client.knowledge_query.return_value = KgQueryResponse(
            agent_id="ag-1", node_count=0, edge_count=0, edges=[]
        )
        helper = DakeraDelegationHelper(client)

        chain = helper.get_delegation_chain("ag-1", "root-only", max_depth=3)

        assert chain == ["root-only"]
