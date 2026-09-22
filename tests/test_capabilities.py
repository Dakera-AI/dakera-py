"""R9 / DAK-10004 — forward-compat: lenient enums + GET /v1/capabilities.

Contract under test (server `routes/capabilities.rs`): every field additive;
unknown fields and unknown strings inside lists MUST be ignored; the SDK must
never fail deserialisation on a model / index kind / search mode / metric /
representation kind / dtype string it does not know.
"""

import copy

import pytest
import responses

from dakera import (
    AsyncDakeraClient,
    BlockDType,
    DakeraClient,
    DistanceMetric,
    EmbeddingModel,
    IndexKind,
    NotFoundError,
    RepresentationKind,
    RoutingMode,
    SearchMode,
    ServerCapabilities,
    UnsupportedCapabilityError,
    ValidationError,
    parse_accepted_values,
)

BASE = "http://localhost:3000"

# A capabilities document as the v0.12 server emits it, PLUS a model string and
# an index kind this SDK does not know, PLUS an extra top-level field, an extra
# model field and an extra records field — all of which must be tolerated.
CAPABILITIES_FIXTURE = {
    "capabilities_version": 1,
    "server_version": "0.12.0",
    "api_versions": ["v1"],
    "default_model": "bge-large",
    "models": [
        {
            "name": "bge-large",
            "aliases": ["bge-large-en", "bge-large-en-v1.5"],
            "dimension": 1024,
            "max_seq_length": 512,
            "effective_max_seq_length": 512,
            "active": True,
            "modality": "text",
        },
        {
            "name": "modernbert-embed-base",
            "aliases": ["modernbert", "modern-bert"],
            "dimension": 768,
            "max_seq_length": 8192,
            "effective_max_seq_length": 2048,
            "active": False,
            "mrl_dimensions": [256, 768],
            "modality": "text",
        },
        {
            "name": "bge-m3",
            "aliases": ["bge-m3-dense", "baai/bge-m3"],
            "dimension": 1024,
            "max_seq_length": 8192,
            "effective_max_seq_length": 2048,
            "active": False,
            "modality": "text",
        },
        {
            # A model from a FUTURE server: not declared in this SDK.
            "name": "colmodernvbert-v9",
            "aliases": [],
            "dimension": 128,
            "max_seq_length": 4096,
            "effective_max_seq_length": 4096,
            "active": False,
            "modality": "image",
            "quantised": True,  # unknown per-model field
        },
    ],
    "index_kinds": ["hnsw", "pq", "ivf", "ivfpq", "spfresh", "fulltext", "muvera_fde"],
    "vector_index_kinds": ["hnsw", "ivf", "ivfpq", "spfresh", "muvera_fde"],
    "live_vector_index_kinds": ["hnsw", "ivf", "spfresh"],
    "distance_metrics": ["cosine", "euclidean", "dot_product", "hamming"],
    "search_mode": "rabitq",
    "search_modes_accepted": "hybrid, binary, float, scalar (alias sq), rabitq, warp9",
    "fulltext_language": "de",
    "on_disk_format_version": 1,
    "records": {
        "enabled": True,
        "representation_kinds": ["dense", "token_multivector", "patch_multivector", "holo"],
        "dtypes": ["f32", "f16", "i8", "e4m3"],
        "max_representations": 8,
        "max_vectors": 4096,
        "max_bytes": 8388608,
        "compression": "zstd",  # unknown field inside records
    },
    "query_languages": ["en", "de", "fr", "es", "it", "pt", "nl"],
    "reembed_pending": True,
    "future_top_level_field": {"anything": [1, 2, 3]},  # unknown top-level field
}


@pytest.fixture
def mock_responses():
    with responses.RequestsMock() as rsps:
        yield rsps


@pytest.fixture
def client():
    return DakeraClient(BASE)


# ============================================================================
# 1. Lenient enum parsing
# ============================================================================


class TestLenientEnums:
    def test_known_values_still_parse_to_declared_members(self):
        assert EmbeddingModel("bge-large") is EmbeddingModel.BGE_LARGE
        assert EmbeddingModel("bge-m3") is EmbeddingModel.BGE_M3
        assert EmbeddingModel.BGE_LARGE.is_known
        assert DistanceMetric("dot_product") is DistanceMetric.DOT_PRODUCT

    @pytest.mark.parametrize(
        "enum_cls, value",
        [
            (EmbeddingModel, "colmodernvbert-v9"),
            (IndexKind, "muvera_fde"),
            (SearchMode, "warp9"),
            (DistanceMetric, "hamming"),
            (RepresentationKind, "holo"),
            (BlockDType, "e4m3"),
            (RoutingMode, "graph"),
        ],
    )
    def test_unknown_value_does_not_raise(self, enum_cls, value):
        member = enum_cls(value)
        assert isinstance(member, enum_cls)
        assert member.value == value
        assert member == value  # str subclass: compares equal to the wire string
        assert not member.is_known
        # Interned: the same unknown string yields the same member.
        assert enum_cls(value) is member
        # Declared members are unaffected.
        assert value not in enum_cls.known_values()
        assert all(enum_cls(v).is_known for v in enum_cls.known_values())

    def test_unknown_member_serialises_back_unchanged(self):
        member = EmbeddingModel("colmodernvbert-v9")
        assert str(member) == "colmodernvbert-v9"
        assert f"{member.value}" == "colmodernvbert-v9"

    def test_non_string_values_still_raise(self):
        with pytest.raises(ValueError):
            EmbeddingModel(42)

    def test_known_values_list_v012_strings(self):
        # The strings v0.12 adds must be declared so users get constants for them.
        assert "bge-m3" in EmbeddingModel.known_values()
        assert "ivfpq" in IndexKind.known_values()
        assert "rabitq" in SearchMode.known_values()
        assert RepresentationKind.known_values() == [
            "dense",
            "token_multivector",
            "patch_multivector",
        ]
        assert BlockDType.known_values() == ["f32", "f16", "i8"]


class TestUnknownEnumInServerResponsesRoundTrips:
    """A response whose `model` names a model this SDK does not know must parse."""

    def test_upsert_text_response_with_unknown_model(self, client, mock_responses):
        mock_responses.add(
            responses.POST,
            f"{BASE}/v1/namespaces/ns/upsert-text",
            json={
                "upserted_count": 1,
                "tokens_processed": 4,
                "model": "colmodernvbert-v9",
                "embedding_time_ms": 3,
                "new_field_from_future": True,
            },
        )
        resp = client.upsert_text("ns", [{"id": "d1", "text": "hi"}])
        assert resp.model == "colmodernvbert-v9"
        assert isinstance(resp.model, EmbeddingModel)
        assert not resp.model.is_known

    def test_query_text_and_batch_with_unknown_model(self, client, mock_responses):
        mock_responses.add(
            responses.POST,
            f"{BASE}/v1/namespaces/ns/query-text",
            json={"results": [], "model": "bge-m3", "embedding_time_ms": 1, "search_time_ms": 1},
        )
        mock_responses.add(
            responses.POST,
            f"{BASE}/v1/namespaces/ns/batch-query-text",
            json={"results": [[]], "model": "x-new", "embedding_time_ms": 1, "search_time_ms": 1},
        )
        assert client.query_text("ns", "q").model is EmbeddingModel.BGE_M3
        batch = client.batch_query_text("ns", ["q"])
        assert batch.model == "x-new" and not batch.model.is_known

    def test_configure_namespace_response_with_unknown_metric(self, client, mock_responses):
        mock_responses.add(
            responses.PUT,
            f"{BASE}/v1/namespaces/ns",
            json={"namespace": "ns", "dimension": 8, "distance": "hamming", "created": True},
        )
        resp = client.configure_namespace("ns", dimension=8)
        assert resp.distance == "hamming"
        assert not resp.distance.is_known

    def test_unknown_model_can_be_sent_as_string(self, client, mock_responses):
        mock_responses.add(
            responses.POST,
            f"{BASE}/v1/namespaces/ns/upsert-text",
            json={"upserted_count": 1, "tokens_processed": 1, "model": "bge-m3"},
        )
        client.upsert_text("ns", [{"id": "d", "text": "t"}], model="bge-m3")
        import json

        sent = json.loads(mock_responses.calls[0].request.body)
        assert sent["model"] == "bge-m3"


# ============================================================================
# 2. GET /v1/capabilities → ServerCapabilities
# ============================================================================


class TestServerCapabilitiesParsing:
    def test_fixture_with_unknown_strings_and_fields_parses(self):
        caps = ServerCapabilities.from_dict(copy.deepcopy(CAPABILITIES_FIXTURE))
        assert caps.capabilities_version == 1
        assert caps.server_version == "0.12.0"
        assert caps.api_versions == ["v1"]
        assert caps.default_model is EmbeddingModel.BGE_LARGE
        assert caps.model_names == [
            "bge-large",
            "modernbert-embed-base",
            "bge-m3",
            "colmodernvbert-v9",
        ]
        unknown = caps.model("colmodernvbert-v9")
        assert unknown is not None and not unknown.name.is_known
        assert unknown.modality == "image"
        assert unknown.raw["quantised"] is True  # unknown per-model field kept, not fatal
        assert caps.model("modernbert").mrl_dimensions == [256, 768]  # alias lookup
        assert caps.model("bge-large").active and caps.active_model.name is EmbeddingModel.BGE_LARGE
        assert IndexKind.IVFPQ in caps.index_kinds
        assert IndexKind("muvera_fde") in caps.index_kinds
        assert caps.live_vector_index_kinds == ["hnsw", "ivf", "spfresh"]
        assert DistanceMetric("hamming") in caps.distance_metrics
        assert caps.search_mode is SearchMode.RABITQ
        assert [m.value for m in caps.search_modes_accepted] == [
            "hybrid",
            "binary",
            "float",
            "scalar",
            "sq",
            "rabitq",
            "warp9",
        ]
        assert caps.fulltext_language == "de"
        assert caps.on_disk_format_version == 1
        assert caps.supports_records and caps.records.enabled
        assert RepresentationKind("holo") in caps.records.representation_kinds
        assert BlockDType("e4m3") in caps.records.dtypes
        assert caps.records.max_bytes == 8 * 1024 * 1024
        assert caps.records.raw["compression"] == "zstd"
        assert caps.query_languages[-1] == "nl"
        assert caps.reembed_pending is True
        assert caps.raw["future_top_level_field"] == {"anything": [1, 2, 3]}

    def test_minimal_and_empty_documents_parse(self):
        # A future server could drop to the bare minimum; a broken proxy could send {}.
        caps = ServerCapabilities.from_dict({})
        assert caps.models == [] and not caps.supports_records and not caps.reembed_pending
        assert caps.search_modes_accepted == []
        caps = ServerCapabilities.from_dict({"models": "not-a-list", "records": None})
        assert caps.models == []

    def test_search_modes_accepted_parser(self):
        assert parse_accepted_values("hybrid, binary, float, scalar (alias sq), rabitq") == [
            "hybrid",
            "binary",
            "float",
            "scalar",
            "sq",
            "rabitq",
        ]
        # Reshaped into a real list one day: also fine.
        assert parse_accepted_values(["hybrid", "float"]) == ["hybrid", "float"]
        assert parse_accepted_values(None) == []
        assert parse_accepted_values("a (alias b, c)") == ["a", "b", "c"]

    def test_supports_helpers(self):
        caps = ServerCapabilities.from_dict(copy.deepcopy(CAPABILITIES_FIXTURE))
        assert caps.supports_model(EmbeddingModel.BGE_M3)
        assert caps.supports_model("baai/bge-m3")  # alias
        assert not caps.supports_model("minilm")
        assert caps.supports_index_kind("ivfpq") and not caps.supports_index_kind("flat")
        assert caps.supports_distance_metric(DistanceMetric.COSINE)
        assert caps.supports_search_mode("sq") and not caps.supports_search_mode("exact")
        assert caps.supports_query_language("de") and not caps.supports_query_language("ja")
        with pytest.raises(ValueError):
            caps.supported_values("nope")


class TestCapabilitiesClient:
    def test_capabilities_is_fetched_once_and_cached(self, client, mock_responses):
        mock_responses.add(responses.GET, f"{BASE}/v1/capabilities", json=CAPABILITIES_FIXTURE)
        first = client.capabilities()
        second = client.capabilities()
        assert first is second
        assert len(mock_responses.calls) == 1
        assert first.reembed_pending is True and first.supports_records

    def test_refresh_refetches(self, client, mock_responses):
        mock_responses.add(responses.GET, f"{BASE}/v1/capabilities", json=CAPABILITIES_FIXTURE)
        changed = copy.deepcopy(CAPABILITIES_FIXTURE)
        changed["reembed_pending"] = False
        mock_responses.add(responses.GET, f"{BASE}/v1/capabilities", json=changed)
        assert client.capabilities().reembed_pending is True
        assert client.capabilities(refresh=True).reembed_pending is False
        assert len(mock_responses.calls) == 2

    def test_pre_012_server_raises_not_found(self, client, mock_responses):
        mock_responses.add(
            responses.GET, f"{BASE}/v1/capabilities", json={"error": "not found"}, status=404
        )
        with pytest.raises(NotFoundError):
            client.capabilities()


# ============================================================================
# 3. Pre-flight validation
# ============================================================================


class TestPreflight:
    def test_unsupported_model_raises_named_error_before_sending(self, client, mock_responses):
        mock_responses.add(responses.GET, f"{BASE}/v1/capabilities", json=CAPABILITIES_FIXTURE)
        client.capabilities()  # populate the cache; no preflight flag needed
        with pytest.raises(UnsupportedCapabilityError) as exc_info:
            client.upsert_text("ns", [{"id": "d", "text": "t"}], model=EmbeddingModel.MINILM)
        err = exc_info.value
        assert isinstance(err, ValidationError)
        assert err.kind == "model" and err.requested == "minilm"
        assert err.supported == [
            "bge-large",
            "modernbert-embed-base",
            "bge-m3",
            "colmodernvbert-v9",
        ]
        assert "minilm" in str(err) and "bge-m3" in str(err) and "v0.12.0" in str(err)
        # Nothing but the capabilities GET went over the wire.
        assert [c.request.method for c in mock_responses.calls] == ["GET"]

    def test_supported_model_and_alias_pass_through(self, client, mock_responses):
        mock_responses.add(responses.GET, f"{BASE}/v1/capabilities", json=CAPABILITIES_FIXTURE)
        mock_responses.add(
            responses.POST,
            f"{BASE}/v1/namespaces/ns/query-text",
            json={"results": [], "model": "bge-m3", "embedding_time_ms": 1, "search_time_ms": 1},
        )
        mock_responses.add(
            responses.POST,
            f"{BASE}/v1/namespaces/ns/batch-query-text",
            json={"results": [], "model": "bge-m3", "embedding_time_ms": 1, "search_time_ms": 1},
        )
        client.capabilities()
        client.query_text("ns", "q", model=EmbeddingModel.BGE_M3)
        client.batch_query_text("ns", ["q"], model="baai/bge-m3")  # alias, plain string
        assert len(mock_responses.calls) == 3

    def test_index_kind_and_distance_metric_preflight(self, client, mock_responses):
        mock_responses.add(responses.GET, f"{BASE}/v1/capabilities", json=CAPABILITIES_FIXTURE)
        client.capabilities()
        with pytest.raises(UnsupportedCapabilityError) as exc_info:
            client.create_namespace("ns", dimensions=8, index_type="flat")
        assert exc_info.value.kind == "index_kind"
        assert "ivfpq" in exc_info.value.supported
        with pytest.raises(UnsupportedCapabilityError) as exc_info:
            client.configure_namespace("ns", dimension=8, distance=DistanceMetric("manhattan"))
        assert exc_info.value.kind == "distance_metric"
        with pytest.raises(UnsupportedCapabilityError):
            client.query("ns", [0.1], distance_metric=DistanceMetric("manhattan"))
        assert len(mock_responses.calls) == 1  # only the capabilities fetch

    def test_require_supported_search_mode(self, client, mock_responses):
        mock_responses.add(responses.GET, f"{BASE}/v1/capabilities", json=CAPABILITIES_FIXTURE)
        client.require_supported("search_mode", SearchMode.RABITQ)
        client.require_supported("search_mode", "sq")  # alias expanded from the prose field
        with pytest.raises(UnsupportedCapabilityError) as exc_info:
            client.require_supported("search_mode", "exact")
        assert exc_info.value.supported == [
            "hybrid",
            "binary",
            "float",
            "scalar",
            "sq",
            "rabitq",
            "warp9",
        ]
        client.require_supported("query_language", "fr")
        with pytest.raises(UnsupportedCapabilityError):
            client.require_supported("query_language", "ja")

    def test_no_preflight_without_cache_by_default(self, client, mock_responses):
        # Default client: capabilities never fetched → the request goes straight out.
        mock_responses.add(
            responses.POST,
            f"{BASE}/v1/namespaces/ns/upsert-text",
            json={"upserted_count": 1, "tokens_processed": 1, "model": "minilm"},
        )
        client.upsert_text("ns", [{"id": "d", "text": "t"}], model=EmbeddingModel.MINILM)
        assert [c.request.method for c in mock_responses.calls] == ["POST"]

    def test_preflight_flag_fetches_lazily(self, mock_responses):
        client = DakeraClient(BASE, preflight=True)
        mock_responses.add(responses.GET, f"{BASE}/v1/capabilities", json=CAPABILITIES_FIXTURE)
        with pytest.raises(UnsupportedCapabilityError):
            client.upsert_text("ns", [{"id": "d", "text": "t"}], model=EmbeddingModel.MINILM)
        assert [c.request.method for c in mock_responses.calls] == ["GET"]

    def test_preflight_flag_degrades_on_pre_012_server(self, mock_responses):
        client = DakeraClient(BASE, preflight=True)
        mock_responses.add(responses.GET, f"{BASE}/v1/capabilities", status=404, json={})
        mock_responses.add(
            responses.POST,
            f"{BASE}/v1/namespaces/ns/upsert-text",
            json={"upserted_count": 1, "tokens_processed": 1, "model": "minilm"},
        )
        mock_responses.add(
            responses.POST,
            f"{BASE}/v1/namespaces/ns/upsert-text",
            json={"upserted_count": 1, "tokens_processed": 1, "model": "minilm"},
        )
        client.upsert_text("ns", [{"id": "d", "text": "t"}], model=EmbeddingModel.MINILM)
        client.upsert_text("ns", [{"id": "d", "text": "t"}], model=EmbeddingModel.MINILM)
        # 404 once, then never asked again for this client.
        assert [c.request.method for c in mock_responses.calls] == ["GET", "POST", "POST"]


class TestAsyncClient:
    @pytest.mark.asyncio
    async def test_async_capabilities_and_preflight(self):
        import httpx

        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(f"{request.method} {request.url.path}")
            if request.url.path == "/v1/capabilities":
                return httpx.Response(200, json=CAPABILITIES_FIXTURE)
            return httpx.Response(
                200,
                json={"upserted_count": 1, "tokens_processed": 1, "model": "colmodernvbert-v9"},
            )

        client = AsyncDakeraClient(BASE, preflight=True)
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=BASE)
        try:
            caps = await client.capabilities()
            assert caps.supports_records and caps.reembed_pending
            assert await client.capabilities() is caps
            with pytest.raises(UnsupportedCapabilityError) as exc_info:
                await client.upsert_text("ns", [{"id": "d", "text": "t"}], model="minilm")
            assert exc_info.value.supported[-1] == "colmodernvbert-v9"
            resp = await client.upsert_text("ns", [{"id": "d", "text": "t"}], model="bge-m3")
            assert resp.model == "colmodernvbert-v9" and not resp.model.is_known
            await client.require_supported("search_mode", "rabitq")
        finally:
            await client.close()
        assert calls == ["GET /v1/capabilities", "POST /v1/namespaces/ns/upsert-text"]
