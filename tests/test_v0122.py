# ruff: noqa: E501
"""Server v0.12.2 support: agents, key edits / rotation grace / whoami, namespace kinds,
capabilities v2, the session lifecycle (idle timeout, touch, ended_reason), opt-in
content previews and ``include_derived`` on listings, derivation status / drain, and
the additive response fields (``session_state``, ``ended_sessions``,
``duplicates_skipped_changed``, ``summaries_skipped``, ``unavailable``).

Wire shapes are taken from the v0.12.2 server source (``routes/agents.rs``,
``routes/keys.rs``, ``routes/namespace_keys.rs``, ``routes/namespaces.rs``,
``routes/sessions.rs``, ``routes/capabilities.rs``, ``derivation/routes.rs``,
``routes/admin``). Every new field must stay optional: the "old server" tests feed
the v0.12.0/v0.12.1 shapes.
"""

import json
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
import responses

from dakera import (
    AgentNetworkNode,
    AgentSummary,
    AsyncChatMemorySession,
    AsyncDakeraClient,
    BatchStoreMemoryItem,
    BatchStoreMemoryRequest,
    BatchStoreMemoryResponse,
    ChatMemorySession,
    CompressResponse,
    ConflictError,
    CreateAgentResponse,
    DakeraClient,
    DeduplicateResponse,
    DerivationDrainResponse,
    DerivationStatus,
    KeyInfo,
    Memory,
    MemoryEvent,
    MemoryTypeStatsResponse,
    NamespaceInfo,
    NamespaceUnavailable,
    RotateKeyResponse,
    ServerCapabilities,
    Session,
    SessionTouchResponse,
    StorageTierOverview,
    TtlStatsResponse,
    WhoamiResponse,
)
from dakera.models import RetryConfig

BASE = "http://localhost:3000"


@pytest.fixture
def rsps():
    with responses.RequestsMock() as r:
        yield r


@pytest.fixture
def client():
    return DakeraClient(BASE, retry_config=RetryConfig(max_retries=1))


def body_of(call):
    return json.loads(call.request.body) if call.request.body else None


def query_of(call):
    return {k: v[0] for k, v in parse_qs(urlparse(call.request.url).query).items()}


class Recorder:
    """httpx handler: answers by (method, path) and records what was sent."""

    def __init__(self, routes):
        self.routes = routes
        self.seen = []

    def __call__(self, request):
        data = json.loads(request.content) if request.content else None
        query = dict(request.url.params.items())
        self.seen.append((request.method, request.url.path, query, data))
        status, payload = self.routes[(request.method, request.url.path)]
        return httpx.Response(status, json=payload)


def _async(handler):
    c = AsyncDakeraClient(BASE, retry_config=RetryConfig(max_retries=1))
    c._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=BASE)
    return c


# ---------------------------------------------------------------------------
# Server shapes (from the v0.12.2 source)
# ---------------------------------------------------------------------------

AGENT_CREATED = {"agent_id": "mlx-dev", "namespace": "_dakera_agent_mlx-dev", "created": True,
                 "dimension": 1024, "model": "bge-large"}  # fmt: skip
KEY_INFO = {"key_id": "dk_key_1a2b3c4d", "name": "dev", "scope": "write",
            "namespaces": ["_dakera_agent_mlx-*"], "created_at": 1791392203, "expires_at": None,
            "active": True, "grants_version": 1, "inert_namespaces": ["foo*", "_dakera_sessions"]}  # fmt: skip
KEY_INFO_V0121 = {"key_id": "dk_key_old", "name": "old", "scope": "read", "namespaces": None,
                  "created_at": 1, "expires_at": None, "active": True}  # fmt: skip
ROTATED = {"new_key": "dk_" + "0" * 32, "key_id": "dk_key_new", "old_key_id": "dk_key_old",
           "old_key_expires_at": 1791395803, "warning": "Save this new key now!"}  # fmt: skip
WHOAMI = {"key_id": "dk_key_1a2b3c4d", "name": "dev", "scope": "write", "namespaces": ["foo*", "docs"],
          "unrestricted": False, "expires_at": None, "grants_version": 0,
          "inert_namespaces": ["foo*"], "auth_enabled": True}  # fmt: skip
WHOAMI_AUTH_OFF = {"key_id": "auth-disabled", "name": "Auth Disabled", "scope": "super_admin",
                   "namespaces": None, "unrestricted": True, "expires_at": None, "grants_version": 1,
                   "inert_namespaces": [], "auth_enabled": False}  # fmt: skip
SESSION = {"id": "sess_1", "agent_id": "mlx-dev", "started_at": 1791392203, "metadata": {"k": "v"},
           "memory_count": 7, "last_activity_at": 1791392803, "idle_timeout_secs": 7200}  # fmt: skip
SESSION_IDLE_ENDED = {"id": "sess_1", "agent_id": "mlx-dev", "started_at": 1791392203,
                      "ended_at": 1791406603, "memory_count": 7, "last_activity_at": 1791392803,
                      "ended_reason": "idle", "idle_since": 1791392803}  # fmt: skip
TOUCH_ACTIVE = {"session": SESSION, "session_state": "active", "idle_deadline_at": 1791400003}
TOUCH_ENDED = {"session": SESSION_IDLE_ENDED, "session_state": "ended"}
CAPS_V2 = {
    "capabilities_version": 2,
    "server_version": "0.12.2",
    "auth": {"prefix_patterns": True, "sessions_by_agent": True, "key_update": True,
             "rotation_grace_max_secs": 604800, "max_grants": 100, "max_grant_len": 255},
    "naming": {"agent_id_pattern": "^[a-zA-Z0-9][a-zA-Z0-9_\\-.]*$", "agent_id_max_bytes": 241,
               "agent_namespace_prefix": "_dakera_agent_", "agent_namespace_max_bytes": 255,
               "namespace_pattern": "^[a-zA-Z0-9][a-zA-Z0-9_-]*$", "namespace_max_bytes": 128,
               "reserved_prefixes": ["_", "system_", "internal_", "admin_"],
               "internal_namespaces": ["_dakera_sessions", "_dakera_embedding_models"],
               "internal_prefixes": ["_dakera_reembed_staging_"]},
    "sessions": {"idle_timeout_secs": 14400, "max_idle_timeout_secs": 2592000, "touch": True,
                 "ended_reason": True},
}  # fmt: skip
DERIVATION_STATUS = {
    "settled": False, "pending_sentences": 4, "pending_parents": 2, "unmarked_parents": 0,
    "stale_children": 1, "orphan_children": 0, "remeta_children": 0, "duplicate_children": 0,
    "legacy_children": 3, "bm25_missing": 0, "graph_owed": 5, "in_flight": 1,
    "graph_queue_owed": 0, "dirty_namespaces": ["_dakera_agent_a"], "namespaces": 3,
    "unreadable_namespaces": [],
    "heal": {"version": 1, "complete": True, "namespace": None, "cursor": None,
             "parents_healed": 12, "graph_adopted": 40, "started_at": 1760000000,
             "completed_at": 1760000010},
    "reconciler": {"state": "sleeping", "last_tick_at": 1760000100, "ticks": 7,
                   "next_namespace": "_dakera_agent_x"},
    "counters": {"derived": 9, "adopted": 2, "bm25_restored": 0},
}  # fmt: skip
SETTLED = {**DERIVATION_STATUS, "settled": True, "pending_sentences": 0, "pending_parents": 0,
           "stale_children": 0, "legacy_children": 0, "graph_owed": 0, "in_flight": 0, "heal": None}  # fmt: skip
DRAIN = {"settled": True, "timed_out": False, "rounds": 2, "elapsed_ms": 1234, "parents_run": 17,
         "pending_left": 0, "deleted": 3, "bm25_restored": 1, "graph_queued": 5, "status": SETTLED}  # fmt: skip
UNAVAILABLE = [{"namespace": "_dakera_agent_x", "reason": "did not answer within 2000 ms"}]
TIERS = {
    "tiers_enabled": True,
    "architecture": [{"name": "hot", "tier_type": "hot", "technology": "mem", "description": "d",
                      "target_latency": "1ms", "capacity": None, "status": "ok", "current_count": 1,
                      "hit_count": 1, "hit_rate": 1.0}],
    "config": {"hot_tier_capacity": 1, "hot_to_warm_threshold_secs": 1,
               "warm_to_cold_threshold_secs": 1, "auto_tier_enabled": True,
               "tier_check_interval_secs": 1},
    "activity": {"promotions": 0, "demotions": 0, "cache_hit_rate": 0.0, "storage_backend": "fs",
                 "promotions_to_hot": 0, "demotions_to_warm": 0, "demotions_to_cold": 0},
}  # fmt: skip
MEMORY_TYPES = {"total": 3, "working": 0, "episodic": 3, "semantic": 0, "procedural": 0,
                "agent_namespaces": 2}  # fmt: skip
TTL = {"namespaces": [], "total_with_ttl": 0, "total_expired": 0}


# ============================================================================
# Agents
# ============================================================================


class TestCreateAgent:
    def test_created_201(self, rsps, client):
        rsps.add(responses.POST, f"{BASE}/v1/agents", json=AGENT_CREATED, status=201)
        out = client.create_agent("mlx-dev")
        assert body_of(rsps.calls[0]) == {"agent_id": "mlx-dev"}
        assert isinstance(out, CreateAgentResponse)
        assert out.created is True and out.namespace == "_dakera_agent_mlx-dev"
        assert out.dimension == 1024 and out.model == "bge-large"

    def test_existing_200_dimension_null(self, rsps, client):
        rsps.add(responses.POST, f"{BASE}/v1/agents",
                 json={**AGENT_CREATED, "created": False, "dimension": None})  # fmt: skip
        out = client.create_agent("mlx-dev")
        assert out.created is False and out.dimension is None

    def test_agent_summary_unavailable(self):
        a = AgentSummary.from_dict({"agent_id": "x", "memory_count": 0, "session_count": 0,
                                    "active_sessions": 0, "vector_count": 0,
                                    "unavailable": "did not answer within 2000 ms"})  # fmt: skip
        assert a.unavailable == "did not answer within 2000 ms" and a.vector_count == 0
        old = AgentSummary.from_dict({"agent_id": "x", "memory_count": 1})
        assert old.unavailable is None and old.vector_count is None


# ============================================================================
# API keys: create, PATCH, rotate grace, whoami, KeyInfo
# ============================================================================


class TestKeys:
    def test_create_key_sends_scope_namespaces_expiry(self, rsps, client):
        rsps.add(responses.POST, f"{BASE}/admin/keys", json={"key_id": "k"})
        client.create_key(
            "dev", scope="write", namespaces=["_dakera_agent_mlx-*"], expires_in_days=7
        )
        assert body_of(rsps.calls[0]) == {"name": "dev", "scope": "write",
                                          "namespaces": ["_dakera_agent_mlx-*"], "expires_in_days": 7}  # fmt: skip

    def test_create_key_defaults_to_read_scope_every_namespace(self, rsps, client):
        rsps.add(responses.POST, f"{BASE}/admin/keys", json={"key_id": "k"})
        client.create_key("dev")
        assert body_of(rsps.calls[0]) == {"name": "dev", "scope": "read"}

    def test_create_key_empty_namespaces_is_sent(self, rsps, client):
        rsps.add(responses.POST, f"{BASE}/admin/keys", json={"key_id": "k"})
        client.create_key("dev", namespaces=[])
        assert body_of(rsps.calls[0])["namespaces"] == []

    def test_update_key_name_only(self, rsps, client):
        rsps.add(responses.PATCH, f"{BASE}/admin/keys/dk_key_1a2b3c4d", json=KEY_INFO)
        out = client.update_key("dk_key_1a2b3c4d", name="renamed")
        assert rsps.calls[0].request.method == "PATCH"
        assert body_of(rsps.calls[0]) == {"name": "renamed"}
        assert isinstance(out, KeyInfo)
        assert out.grants_version == 1 and out.inert_namespaces == ["foo*", "_dakera_sessions"]
        assert out.namespaces == ["_dakera_agent_mlx-*"]

    def test_update_key_namespaces_and_all_namespaces(self, rsps, client):
        rsps.add(responses.PATCH, f"{BASE}/admin/keys/k1", json=KEY_INFO)
        rsps.add(responses.PATCH, f"{BASE}/admin/keys/k1", json={**KEY_INFO, "namespaces": None})
        rsps.add(responses.PATCH, f"{BASE}/admin/keys/k1", json={**KEY_INFO, "namespaces": []})
        client.update_key("k1", name="n", namespaces=["team-*", "docs"])
        out = client.update_key("k1", all_namespaces=True)
        none = client.update_key("k1", namespaces=[])
        assert body_of(rsps.calls[0]) == {"name": "n", "namespaces": ["team-*", "docs"]}
        assert body_of(rsps.calls[1]) == {"namespaces": None}  # explicit null = every namespace
        assert body_of(rsps.calls[2]) == {"namespaces": []}
        assert out.namespaces is None and none.namespaces == []

    def test_update_key_refuses_empty_or_contradictory(self, client):
        with pytest.raises(ValueError):
            client.update_key("k1")
        with pytest.raises(ValueError):
            client.update_key("k1", namespaces=["a"], all_namespaces=True)

    def test_update_namespace_key(self, rsps, client):
        rsps.add(responses.PATCH, f"{BASE}/v1/namespaces/team-a/keys/k1", json=KEY_INFO)
        out = client.update_namespace_key("team-a", "k1", namespaces=["team-a*"])
        assert body_of(rsps.calls[0]) == {"namespaces": ["team-a*"]}
        assert out.key_id == "dk_key_1a2b3c4d"

    def test_update_namespace_key_conflict_on_inactive(self, rsps, client):
        rsps.add(responses.PATCH, f"{BASE}/v1/namespaces/n/keys/k1",
                 json={"error": "inactive", "code": "CONFLICT"}, status=409)  # fmt: skip
        with pytest.raises(ConflictError):
            client.update_namespace_key("n", "k1", name="x")

    def test_rotate_with_grace(self, rsps, client):
        rsps.add(responses.POST, f"{BASE}/admin/keys/dk_key_old/rotate", json=ROTATED)
        out = client.rotate_key("dk_key_old", grace_secs=3600)
        assert body_of(rsps.calls[0]) == {"grace_secs": 3600}
        assert out["old_key_id"] == "dk_key_old" and out["old_key_expires_at"] == 1791395803
        typed = RotateKeyResponse.from_dict(out)
        assert typed.key_id == "dk_key_new" and typed.old_key_expires_at == 1791395803

    def test_rotate_without_grace_sends_no_body(self, rsps, client):
        old = {"new_key": "dk_x", "key_id": "dk_key_new", "warning": "w"}  # v0.12.1 shape
        rsps.add(responses.POST, f"{BASE}/admin/keys/k/rotate", json=old)
        out = client.rotate_key("k")
        assert rsps.calls[0].request.body is None
        typed = RotateKeyResponse.from_dict(out)
        assert typed.old_key_id is None and typed.old_key_expires_at is None

    def test_whoami(self, rsps, client):
        rsps.add(responses.GET, f"{BASE}/v1/auth/whoami", json=WHOAMI)
        rsps.add(responses.GET, f"{BASE}/v1/auth/whoami", json=WHOAMI_AUTH_OFF)
        me = client.whoami()
        assert isinstance(me, WhoamiResponse)
        assert me.grants_version == 0 and me.inert_namespaces == ["foo*"]
        assert me.unrestricted is False and me.auth_enabled is True
        off = client.whoami()
        assert off.auth_enabled is False and off.namespaces is None and off.unrestricted

    def test_key_info_from_older_server(self):
        k = KeyInfo.from_dict(KEY_INFO_V0121)
        assert k.grants_version is None and k.inert_namespaces == [] and k.namespaces is None


class TestNamespaceKeys:
    CREATED = {"key_id": "k1", "key": "dk_" + "1" * 32, "name": "ci", "scope": "write",
               "namespaces": ["team-a", "team-a-docs*"], "created_at": 5, "expires_at": None,
               "warning": "Save this key"}  # fmt: skip

    def test_create_sends_scope_and_extra_namespaces(self, rsps, client):
        rsps.add(responses.POST, f"{BASE}/v1/namespaces/team-a/keys", json=self.CREATED)
        out = client.create_namespace_key(
            "team-a", "ci", scope="write", extra_namespaces=["team-a-docs*"]
        )
        assert body_of(rsps.calls[0]) == {"name": "ci", "scope": "write",
                                          "extra_namespaces": ["team-a-docs*"]}  # fmt: skip
        # The server's answer has no "namespace" field: the path namespace is used.
        assert out.namespace == "team-a" and out.scope == "write"
        assert out.namespaces == ["team-a", "team-a-docs*"]

    def test_create_default_scope(self, rsps, client):
        rsps.add(responses.POST, f"{BASE}/v1/namespaces/n/keys", json=self.CREATED)
        client.create_namespace_key("n", "ci", expires_in_days=3)
        assert body_of(rsps.calls[0]) == {"name": "ci", "scope": "read", "expires_in_days": 3}

    def test_list_parses_server_shape(self, rsps, client):
        rsps.add(responses.GET, f"{BASE}/v1/namespaces/team-a/keys",
                 json={"keys": [KEY_INFO, KEY_INFO_V0121], "total": 2})  # fmt: skip
        out = client.list_namespace_keys("team-a")
        assert out.namespace == "team-a" and out.total == 2
        assert out.keys[0].namespace == "team-a" and out.keys[0].grants_version == 1
        assert out.keys[0].inert_namespaces == ["foo*", "_dakera_sessions"]
        assert out.keys[1].grants_version is None and out.keys[1].inert_namespaces == []

    def test_usage_parses_server_shape(self, rsps, client):
        rsps.add(responses.GET, f"{BASE}/v1/namespaces/n/keys/k1/usage",
                 json={"key_id": "k1", "total_requests": 3, "successful_requests": 3,
                       "failed_requests": 0, "rate_limited_requests": 0, "bytes_transferred": 9,
                       "avg_latency_ms": 1.5, "by_endpoint": [], "by_namespace": []})  # fmt: skip
        out = client.get_namespace_key_usage("n", "k1")
        assert out.namespace == "n" and out.total_requests == 3


# ============================================================================
# Namespace kinds, capabilities v2
# ============================================================================


class TestNamespaceKinds:
    def test_list_with_kinds(self, rsps, client):
        rsps.add(responses.GET, f"{BASE}/v1/namespaces",
                 json={"namespaces": ["docs", "_dakera_agent_a"],
                       "kinds": {"docs": "data", "_dakera_agent_a": "agent"}})  # fmt: skip
        out = client.list_namespaces()
        assert [(n.name, n.kind) for n in out] == [("docs", "data"), ("_dakera_agent_a", "agent")]

    def test_list_older_server_without_kinds(self, rsps, client):
        rsps.add(responses.GET, f"{BASE}/v1/namespaces", json={"namespaces": ["docs"]})
        out = client.list_namespaces()
        assert out[0].name == "docs" and out[0].kind is None

    def test_get_namespace_kind(self, rsps, client):
        rsps.add(responses.GET, f"{BASE}/v1/namespaces/_dakera_agent_a",
                 json={"namespace": "_dakera_agent_a", "kind": "agent", "vector_count": 3,
                       "dimension": 1024})  # fmt: skip
        assert client.get_namespace("_dakera_agent_a").kind == "agent"

    def test_list_from_response_objects_take_kind_from_map(self):
        out = NamespaceInfo.list_from_response(
            {"namespaces": [{"name": "a", "vector_count": 1}], "kinds": {"a": "data"}}
        )
        assert out[0].kind == "data"
        assert NamespaceInfo.list_from_response(None) == []


class TestCapabilitiesV2:
    def test_blocks_parse(self):
        caps = ServerCapabilities.from_dict(CAPS_V2)
        assert caps.capabilities_version == 2
        assert caps.auth.prefix_patterns and caps.auth.rotation_grace_max_secs == 604800
        assert caps.auth.max_grants == 100 and caps.auth.max_grant_len == 255
        assert caps.naming.agent_id_max_bytes == 241
        assert caps.naming.agent_namespace_prefix == "_dakera_agent_"
        assert "_dakera_embedding_models" in caps.naming.internal_namespaces
        assert caps.sessions.idle_timeout_secs == 14400 and caps.sessions.touch
        assert caps.sessions.max_idle_timeout_secs == 2592000 and caps.sessions.ended_reason
        assert caps.supports_prefix_grants and caps.supports_key_update
        assert caps.supports_session_touch

    def test_v1_document_defaults(self):
        caps = ServerCapabilities.from_dict({"capabilities_version": 1, "server_version": "0.12.1"})
        assert caps.auth.prefix_patterns is False and caps.naming.agent_id_max_bytes == 0
        assert caps.sessions.idle_timeout_secs is None and not caps.supports_session_touch

    def test_capabilities_call(self, rsps, client):
        rsps.add(responses.GET, f"{BASE}/v1/capabilities", json=CAPS_V2)
        assert client.capabilities().sessions.idle_timeout_secs == 14400


# ============================================================================
# Sessions lifecycle
# ============================================================================


class TestSessions:
    def test_start_with_idle_timeout(self, rsps, client):
        rsps.add(responses.POST, f"{BASE}/v1/sessions/start", json={"session": SESSION})
        out = client.start_session("mlx-dev", idle_timeout_secs=7200)
        assert body_of(rsps.calls[0]) == {"agent_id": "mlx-dev", "idle_timeout_secs": 7200}
        assert out["idle_timeout_secs"] == 7200

    def test_start_omits_idle_timeout_by_default_and_sends_zero(self, rsps, client):
        rsps.add(responses.POST, f"{BASE}/v1/sessions/start", json={"session": SESSION})
        rsps.add(responses.POST, f"{BASE}/v1/sessions/start", json={"session": SESSION})
        client.start_session("mlx-dev")
        client.start_session("mlx-dev", idle_timeout_secs=0)
        assert body_of(rsps.calls[0]) == {"agent_id": "mlx-dev"}
        assert body_of(rsps.calls[1]) == {"agent_id": "mlx-dev", "idle_timeout_secs": 0}

    def test_touch_active(self, rsps, client):
        rsps.add(responses.POST, f"{BASE}/v1/sessions/sess_1/touch", json=TOUCH_ACTIVE)
        out = client.touch_session("sess_1")
        assert rsps.calls[0].request.body is None
        assert isinstance(out, SessionTouchResponse) and out.is_active
        assert out.idle_deadline_at == 1791400003
        assert out.session.session_id == "sess_1" and out.session.last_activity_at == 1791392803

    def test_touch_ended(self, rsps, client):
        rsps.add(responses.POST, f"{BASE}/v1/sessions/sess_1/touch", json=TOUCH_ENDED)
        out = client.touch_session("sess_1")
        assert out.session_state == "ended" and not out.is_active
        assert out.idle_deadline_at is None
        assert out.session.ended_reason == "idle" and out.session.idle_since == 1791392803
        assert out.session.is_ended

    def test_session_model_new_and_old_shapes(self):
        s = Session.from_dict(SESSION)
        assert s.session_id == "sess_1" and s.idle_timeout_secs == 7200 and not s.is_ended
        old = Session.from_dict({"id": "s", "agent_id": "a", "started_at": 1, "ended_at": 2})
        assert old.ended_reason is None and old.last_activity_at is None and old.is_ended
        legacy = Session.from_dict({"session_id": "s", "agent_id": "a"})
        assert legacy.session_id == "s"

    def test_store_reports_session_state(self, rsps, client):
        memory = {"id": "m1", "content": "x", "agent_id": "a"}
        rsps.add(responses.POST, f"{BASE}/v1/memory/store",
                 json={"memory": memory, "embedding_time_ms": 3, "session_state": "ended"})  # fmt: skip
        rsps.add(responses.POST, f"{BASE}/v1/memory/store",
                 json={"memory": memory, "embedding_time_ms": 3})  # fmt: skip
        assert client.store_memory("a", "x", session_id="sess_1")["session_state"] == "ended"
        assert "session_state" not in client.store_memory("a", "x")

    def test_batch_ended_sessions(self, rsps, client):
        rsps.add(responses.POST, f"{BASE}/v1/memories/store/batch",
                 json={"stored": [], "stored_count": 0, "total_embedding_time_ms": 1,
                       "ended_sessions": ["sess_a"]})  # fmt: skip
        out = client.store_memories_batch(BatchStoreMemoryRequest("a", [BatchStoreMemoryItem("x")]))
        assert out.ended_sessions == ["sess_a"]
        old = BatchStoreMemoryResponse.from_dict({"stored": [], "stored_count": 0})
        assert old.ended_sessions == []

    def test_update_config_session_idle_timeout(self, rsps, client):
        rsps.add(responses.PUT, f"{BASE}/v1/admin/config", json={"success": True})
        rsps.add(responses.PUT, f"{BASE}/v1/admin/config", json={"success": True})
        client.update_config(session_idle_timeout_secs=3600)
        client.update_config({"cache_enabled": False}, session_idle_timeout_secs=0)
        assert body_of(rsps.calls[0]) == {"session_idle_timeout_secs": 3600}
        assert body_of(rsps.calls[1]) == {"cache_enabled": False, "session_idle_timeout_secs": 0}

    def test_update_config_dict_unchanged(self, rsps, client):
        rsps.add(responses.PUT, f"{BASE}/v1/admin/config", json={"success": True})
        client.update_config({"rate_limit_rps": 50})
        assert body_of(rsps.calls[0]) == {"rate_limit_rps": 50}

    def test_session_ended_event_reason(self):
        e = MemoryEvent.from_dict({"event_type": "session_ended", "agent_id": "a",
                                   "session_id": "s", "timestamp": 1, "reason": "idle"})  # fmt: skip
        assert e.reason == "idle"
        assert MemoryEvent.from_dict({"event_type": "stored", "timestamp": 1}).reason is None

    def test_chat_session_idle_timeout_and_touch(self, rsps, client):
        rsps.add(responses.POST, f"{BASE}/v1/sessions/start", json={"session": SESSION})
        rsps.add(responses.POST, f"{BASE}/v1/sessions/sess_1/touch", json=TOUCH_ACTIVE)
        s = ChatMemorySession.create(client, "mlx-dev", idle_timeout_secs=600)
        assert body_of(rsps.calls[0])["idle_timeout_secs"] == 600
        assert s.touch().is_active


# ============================================================================
# Listings: include_derived, content previews
# ============================================================================


class TestListings:
    ROW = {"id": "m1", "content": "abc", "content_len": 90000, "content_truncated": True,
           "memory_type": "Episodic", "importance": 0.5}  # fmt: skip

    def test_agent_memories_params(self, rsps, client):
        rsps.add(responses.GET, f"{BASE}/v1/agents/a/memories", json=[self.ROW])
        out = client.agent_memories("a", limit=10, offset=20, include_derived=True,
                                    content_preview_chars=200)  # fmt: skip
        assert query_of(rsps.calls[0]) == {"limit": "10", "offset": "20", "include_derived": "true",
                                           "content_preview_chars": "200"}  # fmt: skip
        assert out[0]["content_truncated"] is True and out[0]["content_len"] == 90000

    def test_agent_memories_sends_nothing_new_by_default(self, rsps, client):
        rsps.add(responses.GET, f"{BASE}/v1/agents/a/memories", json=[])
        client.agent_memories("a")
        assert query_of(rsps.calls[0]) == {}

    def test_session_memories_params(self, rsps, client):
        rsps.add(responses.GET, f"{BASE}/v1/sessions/s/memories",
                 json={"session": SESSION, "memories": [self.ROW], "total": 1})  # fmt: skip
        rsps.add(responses.GET, f"{BASE}/v1/sessions/s/memories",
                 json={"session": SESSION, "memories": [], "total": 0})  # fmt: skip
        out = client.session_memories("s", limit=5, offset=1, content_preview_chars=80)
        client.session_memories("s")
        assert query_of(rsps.calls[0]) == {
            "limit": "5",
            "offset": "1",
            "content_preview_chars": "80",
        }
        assert query_of(rsps.calls[1]) == {}
        assert out[0]["content_len"] == 90000

    def test_wake_up_include_derived(self, rsps, client):
        rsps.add(responses.GET, f"{BASE}/v1/agents/a/wake-up",
                 json={"agent_id": "a", "memories": [self.ROW], "total_available": 1})  # fmt: skip
        rsps.add(responses.GET, f"{BASE}/v1/agents/a/wake-up",
                 json={"agent_id": "a", "memories": [], "total_available": 0})  # fmt: skip
        out = client.wake_up("a", include_derived=True)
        client.wake_up("a")
        assert query_of(rsps.calls[0])["include_derived"] == "true"
        assert "include_derived" not in query_of(rsps.calls[1])
        assert out.memories[0].content_truncated is True and out.memories[0].content_len == 90000

    def test_memory_without_preview_fields(self):
        m = Memory.from_dict({"id": "m", "content": "x"})
        assert m.content_len is None and m.content_truncated is None

    def test_full_graph_preview(self, rsps, client):
        rsps.add(responses.POST, f"{BASE}/v1/knowledge/graph/full",
                 json={"nodes": [{"id": "m1", "content": "ab", "content_len": 5,
                                  "content_truncated": True}], "edges": [], "clusters": [],
                       "stats": {}})  # fmt: skip
        rsps.add(responses.POST, f"{BASE}/v1/knowledge/graph/full", json={"nodes": []})
        out = client.full_knowledge_graph("a", max_nodes=50, content_preview_chars=200)
        client.full_knowledge_graph("a")
        assert body_of(rsps.calls[0]) == {
            "agent_id": "a",
            "max_nodes": 50,
            "content_preview_chars": 200,
        }
        assert body_of(rsps.calls[1]) == {"agent_id": "a"}
        assert out["nodes"][0]["content_truncated"] is True

    def test_cross_agent_network_preview(self, rsps, client):
        node = {"id": "m", "agent_id": "a", "content": "ab", "content_len": 9,
                "content_truncated": True, "importance": 0.7, "tags": [], "memory_type": "Semantic",
                "created_at": 1}  # fmt: skip
        stats = {"total_agents": 1, "total_nodes": 1, "total_cross_edges": 0, "density": 0.0}
        rsps.add(responses.POST, f"{BASE}/v1/knowledge/network/cross-agent",
                 json={"agents": [], "nodes": [node], "edges": [], "stats": stats})  # fmt: skip
        out = client.cross_agent_network(content_preview_chars=200)
        assert body_of(rsps.calls[0])["content_preview_chars"] == 200
        assert out.nodes[0].content_len == 9 and out.nodes[0].content_truncated is True

    def test_cross_agent_network_omits_preview_and_parses_old_nodes(self, rsps, client):
        stats = {"total_agents": 0, "total_nodes": 0, "total_cross_edges": 0, "density": 0.0}
        rsps.add(responses.POST, f"{BASE}/v1/knowledge/network/cross-agent",
                 json={"agents": [], "nodes": [], "edges": [], "stats": stats})  # fmt: skip
        client.cross_agent_network()
        assert "content_preview_chars" not in body_of(rsps.calls[0])
        old = AgentNetworkNode.from_dict({"id": "m", "agent_id": "a", "content": "x",
                                          "importance": 0.5, "memory_type": "Semantic",
                                          "created_at": 1})  # fmt: skip
        assert old.content_len is None and old.content_truncated is None


# ============================================================================
# Derived data: status / drain
# ============================================================================


class TestDerivations:
    def test_status(self, rsps, client):
        rsps.add(responses.GET, f"{BASE}/v1/admin/derivations/status", json=DERIVATION_STATUS)
        st = client.derivations_status()
        assert isinstance(st, DerivationStatus) and not st.settled
        assert st.pending_sentences == 4 and st.graph_owed == 5 and st.legacy_children == 3
        assert st.dirty_namespaces == ["_dakera_agent_a"] and st.namespaces == 3
        assert st.heal is not None and st.heal.complete and st.heal.parents_healed == 12
        assert st.reconciler.state == "sleeping" and st.reconciler.ticks == 7
        assert st.counters["derived"] == 9

    def test_drain_with_timeout(self, rsps, client):
        rsps.add(responses.POST, f"{BASE}/v1/admin/derivations/drain", json=DRAIN)
        out = client.drain_derivations(timeout_secs=120)
        assert body_of(rsps.calls[0]) == {"timeout_secs": 120}
        assert isinstance(out, DerivationDrainResponse)
        assert out.settled and not out.timed_out and out.rounds == 2 and out.graph_queued == 5
        assert out.status.settled and out.status.heal is None

    def test_drain_default_sends_no_body(self, rsps, client):
        rsps.add(responses.POST, f"{BASE}/v1/admin/derivations/drain", json=DRAIN)
        client.drain_derivations()
        assert rsps.calls[0].request.body is None

    def test_drain_already_running_is_409(self, rsps, client):
        rsps.add(responses.POST, f"{BASE}/v1/admin/derivations/drain",
                 json={"error": "a drain is running", "code": "CONFLICT"}, status=409)  # fmt: skip
        with pytest.raises(ConflictError):
            client.drain_derivations()


# ============================================================================
# Additive response fields: dedup, compress, unavailable
# ============================================================================


class TestAdditiveFields:
    def test_dedup_skipped_changed(self, rsps, client):
        payload = {"groups": [], "duplicates_found": 3, "duplicates_merged": 2,
                   "duplicates_skipped_changed": 1}  # fmt: skip
        rsps.add(responses.POST, f"{BASE}/v1/knowledge/deduplicate", json=payload)
        out = client.deduplicate("a")
        assert out["duplicates_skipped_changed"] == 1
        typed = DeduplicateResponse.from_dict(out)
        assert typed.duplicates_skipped_changed == 1 and typed.duplicates_merged == 2
        assert (
            DeduplicateResponse.from_dict({"duplicates_found": 0}).duplicates_skipped_changed
            is None
        )

    def test_compress_summaries_skipped(self, rsps, client):
        payload = {"agent_id": "a", "memories_scanned": 40, "clusters_found": 3,
                   "summaries_created": 2, "originals_deprecated": 9,
                   "summary_ids": ["mem_compress_1"], "deprecated_ids": ["m1"],
                   "summaries_skipped": [{"summary_id": "mem_compress_2",
                                          "reason": "content: content exceeds maximum of 100000 bytes"}]}  # fmt: skip
        rsps.add(responses.POST, f"{BASE}/v1/agents/a/compress", json=payload)
        out = client.compress_agent("a")
        assert isinstance(out, CompressResponse)
        assert out.memories_scanned == 40 and out.summaries_created == 2
        assert out.summary_ids == ["mem_compress_1"] and out.deprecated_ids == ["m1"]
        assert out.summaries_skipped[0].summary_id == "mem_compress_2"
        assert "bytes" in out.summaries_skipped[0].reason

    def test_compress_older_server(self):
        out = CompressResponse.from_dict({"agent_id": "a", "memories_scanned": 1})
        assert out.summaries_skipped == [] and out.clusters_found is None

    def test_unavailable_on_typed_node_wide_responses(self, rsps, client):
        rsps.add(
            responses.GET, f"{BASE}/v1/admin/ttl/stats", json={**TTL, "unavailable": UNAVAILABLE}
        )
        rsps.add(responses.GET, f"{BASE}/v1/admin/memory-type-stats",
                 json={**MEMORY_TYPES, "unavailable": UNAVAILABLE})  # fmt: skip
        rsps.add(
            responses.GET,
            f"{BASE}/v1/admin/storage/tiers",
            json={**TIERS, "unavailable": UNAVAILABLE},
        )
        for out in (client.ttl_stats(), client.memory_type_stats(), client.storage_tier_overview()):
            assert out.unavailable == [
                NamespaceUnavailable("_dakera_agent_x", "did not answer within 2000 ms")
            ]

    def test_unavailable_absent(self):
        assert TtlStatsResponse.from_dict(TTL).unavailable == []
        assert MemoryTypeStatsResponse.from_dict(MEMORY_TYPES).unavailable == []
        assert StorageTierOverview.from_dict(TIERS).unavailable == []
        assert NamespaceUnavailable.list_from(None) == []

    def test_ops_stats_passes_unavailable_through(self, rsps, client):
        rsps.add(responses.GET, f"{BASE}/v1/ops/stats",
                 json={"total_vectors": 1, "namespace_count": 2, "unavailable": UNAVAILABLE})  # fmt: skip
        assert client.ops_stats()["unavailable"] == UNAVAILABLE


# ============================================================================
# Async client
# ============================================================================


class TestAsync:
    async def test_agents_keys_whoami(self):
        rec = Recorder({
            ("POST", "/v1/agents"): (201, AGENT_CREATED),
            ("PATCH", "/admin/keys/k1"): (200, KEY_INFO),
            ("PATCH", "/v1/namespaces/team-a/keys/k1"): (200, KEY_INFO),
            ("POST", "/admin/keys/k1/rotate"): (200, ROTATED),
            ("GET", "/v1/auth/whoami"): (200, WHOAMI),
            ("POST", "/admin/keys"): (201, {"key_id": "k"}),
        })  # fmt: skip
        c = _async(rec)
        assert (await c.create_agent("mlx-dev")).created
        assert (await c.update_key("k1", all_namespaces=True)).grants_version == 1
        assert (await c.update_namespace_key("team-a", "k1", name="x")).key_id == "dk_key_1a2b3c4d"
        assert (await c.rotate_key("k1", grace_secs=60))["old_key_id"] == "dk_key_old"
        assert (await c.whoami()).inert_namespaces == ["foo*"]
        await c.create_key("dev", scope="admin", namespaces=["team-*"])
        with pytest.raises(ValueError):
            await c.update_key("k1")
        bodies = [d for _, _, _, d in rec.seen]
        assert bodies == [
            {"agent_id": "mlx-dev"},
            {"namespaces": None},
            {"name": "x"},
            {"grace_secs": 60},
            None,
            {"name": "dev", "scope": "admin", "namespaces": ["team-*"]},
        ]

    async def test_namespace_keys_and_kinds(self):
        rec = Recorder({
            ("POST", "/v1/namespaces/team-a/keys"): (201, TestNamespaceKeys.CREATED),
            ("GET", "/v1/namespaces/team-a/keys"): (200, {"keys": [KEY_INFO], "total": 1}),
            ("GET", "/v1/namespaces"): (200, {"namespaces": ["docs"], "kinds": {"docs": "data"}}),
        })  # fmt: skip
        c = _async(rec)
        created = await c.create_namespace_key("team-a", "ci", extra_namespaces=["team-a-x*"])
        assert created.namespace == "team-a"
        listed = await c.list_namespace_keys("team-a")
        assert listed.keys[0].namespace == "team-a" and listed.keys[0].grants_version == 1
        assert [(n.name, n.kind) for n in await c.list_namespaces()] == [("docs", "data")]
        assert rec.seen[0][3] == {"name": "ci", "scope": "read", "extra_namespaces": ["team-a-x*"]}

    async def test_sessions(self):
        rec = Recorder({
            ("POST", "/v1/sessions/start"): (200, {"session": SESSION}),
            ("POST", "/v1/sessions/sess_1/touch"): (200, TOUCH_ENDED),
            ("GET", "/v1/sessions/sess_1/memories"): (200, {"session": SESSION, "memories": [{"id": "m"}], "total": 1}),
            ("POST", "/v1/memory/store"): (200, {"memory": {"id": "m", "content": "x"}, "session_state": "active"}),
            ("POST", "/v1/memories/store/batch"): (200, {"stored": [], "stored_count": 0, "ended_sessions": ["s"]}),
            ("PUT", "/v1/admin/config"): (200, {"success": True}),
        })  # fmt: skip
        c = _async(rec)
        await c.start_session("mlx-dev", idle_timeout_secs=0)
        touched = await c.touch_session("sess_1")
        assert touched.session_state == "ended" and touched.session.ended_reason == "idle"
        mems = await c.session_memories("sess_1", content_preview_chars=100)
        assert mems == [{"id": "m"}]  # the memories list, as in the sync client
        assert (await c.store_memory("a", "x", session_id="sess_1"))["session_state"] == "active"
        batch = await c.store_memories_batch(
            BatchStoreMemoryRequest("a", [BatchStoreMemoryItem("x")])
        )
        assert batch.ended_sessions == ["s"]
        await c.update_config(session_idle_timeout_secs=60)
        assert rec.seen[0][3] == {"agent_id": "mlx-dev", "idle_timeout_secs": 0}
        assert rec.seen[1][3] is None
        assert rec.seen[2][2] == {"content_preview_chars": "100"}
        assert rec.seen[5][3] == {"session_idle_timeout_secs": 60}

    async def test_chat_session_touch(self):
        rec = Recorder({
            ("POST", "/v1/sessions/start"): (200, {"session": SESSION}),
            ("POST", "/v1/sessions/sess_1/touch"): (200, TOUCH_ACTIVE),
        })  # fmt: skip
        c = _async(rec)
        s = await AsyncChatMemorySession.create(c, "mlx-dev", idle_timeout_secs=30)
        assert (await s.touch()).idle_deadline_at == 1791400003
        assert rec.seen[0][3]["idle_timeout_secs"] == 30

    async def test_listings_and_graphs(self):
        stats = {"total_agents": 0, "total_nodes": 0, "total_cross_edges": 0, "density": 0.0}
        rec = Recorder({
            ("GET", "/v1/agents/a/memories"): (200, [TestListings.ROW]),
            ("GET", "/v1/agents/a/wake-up"): (200, {"agent_id": "a", "memories": [], "total_available": 0}),
            ("POST", "/v1/knowledge/graph/full"): (200, {"nodes": []}),
            ("POST", "/v1/knowledge/network/cross-agent"): (200, {"agents": [], "nodes": [], "edges": [], "stats": stats}),
        })  # fmt: skip
        c = _async(rec)
        await c.agent_memories("a", offset=5, include_derived=False, content_preview_chars=10)
        await c.wake_up("a", include_derived=True)
        await c.full_knowledge_graph("a", content_preview_chars=200)
        await c.cross_agent_network(content_preview_chars=300)
        assert rec.seen[0][2] == {
            "offset": "5",
            "include_derived": "false",
            "content_preview_chars": "10",
        }
        assert rec.seen[1][2]["include_derived"] == "true"
        assert rec.seen[2][3] == {"agent_id": "a", "content_preview_chars": 200}
        assert rec.seen[3][3]["content_preview_chars"] == 300

    async def test_derivations(self):
        rec = Recorder({
            ("GET", "/v1/admin/derivations/status"): (200, DERIVATION_STATUS),
            ("POST", "/v1/admin/derivations/drain"): (200, DRAIN),
        })  # fmt: skip
        c = _async(rec)
        assert (await c.derivations_status()).pending_parents == 2
        out = await c.drain_derivations(timeout_secs=30)
        assert out.settled and out.parents_run == 17
        assert rec.seen[1][3] == {"timeout_secs": 30}

    async def test_drain_conflict(self):
        rec = Recorder(
            {("POST", "/v1/admin/derivations/drain"): (409, {"error": "busy", "code": "CONFLICT"})}
        )
        with pytest.raises(ConflictError):
            await _async(rec).drain_derivations()
