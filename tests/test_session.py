"""Tests for ChatMemorySession helper."""

from unittest.mock import MagicMock, patch

import pytest
import responses

from dakera import DakeraClient
from dakera.models import RecallResponse, RecalledMemory
from dakera.session import ChatMemorySession


@pytest.fixture
def client():
    return DakeraClient("http://localhost:3000")


@pytest.fixture
def mock_responses():
    with responses.RequestsMock() as rsps:
        yield rsps


class TestChatMemorySessionCreate:
    def test_create_starts_session(self, client, mock_responses):
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/sessions/start",
            json={"session": {"id": "sess_abc123", "agent_id": "agent-1"}},
            status=200,
        )

        session = ChatMemorySession.create(client, "agent-1")

        assert session.session_id == "sess_abc123"
        assert session.agent_id == "agent-1"
        assert len(mock_responses.calls) == 1

    def test_create_with_metadata(self, client, mock_responses):
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/sessions/start",
            json={"session": {"id": "sess_xyz", "agent_id": "agent-2"}},
            status=200,
        )

        session = ChatMemorySession.create(
            client, "agent-2", metadata={"source": "playground"}
        )

        assert session.session_id == "sess_xyz"
        body = mock_responses.calls[0].request.body
        import json
        parsed = json.loads(body)
        assert parsed["metadata"] == {"source": "playground"}


class TestChatMemorySessionStore:
    def test_store_attaches_session_and_role_tag(self, client, mock_responses):
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/memory/store",
            json={"id": "mem_1", "content": "Hello", "agent_id": "agent-1"},
            status=200,
        )

        session = ChatMemorySession(client, "agent-1", "sess_abc")
        session.store("user", "Hello, I am Alice.")

        import json
        body = json.loads(mock_responses.calls[0].request.body)
        assert body["session_id"] == "sess_abc"
        assert body["agent_id"] == "agent-1"
        assert "user" in body["tags"]
        assert body["memory_type"] == "episodic"

    def test_store_custom_importance(self, client, mock_responses):
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/memory/store",
            json={"id": "mem_2", "content": "answer"},
            status=200,
        )

        session = ChatMemorySession(client, "agent-1", "sess_abc")
        session.store("assistant", "The answer is 42.", importance=0.8)

        import json
        body = json.loads(mock_responses.calls[0].request.body)
        assert body["importance"] == 0.8

    def test_store_does_not_duplicate_role_tag(self, client, mock_responses):
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/memory/store",
            json={"id": "mem_3"},
            status=200,
        )

        session = ChatMemorySession(client, "agent-1", "sess_abc")
        session.store("user", "repeat tag test", tags=["user", "important"])

        import json
        body = json.loads(mock_responses.calls[0].request.body)
        assert body["tags"].count("user") == 1


class TestChatMemorySessionRecall:
    def test_recall_returns_memories(self, client, mock_responses):
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/memory/recall",
            json={
                "memories": [
                    {
                        "memory": {
                            "id": "mem_1",
                            "content": "Alice likes Python",
                            "agent_id": "agent-1",
                            "memory_type": "episodic",
                            "importance": 0.6,
                            "created_at": 1781000000,
                        },
                        "score": 0.9,
                    }
                ]
            },
            status=200,
        )

        session = ChatMemorySession(client, "agent-1", "sess_abc")
        memories = session.recall("user preferences")

        assert len(memories) == 1
        assert memories[0].content == "Alice likes Python"

    def test_recall_with_custom_top_k(self, client, mock_responses):
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/memory/recall",
            json={"memories": []},
            status=200,
        )

        session = ChatMemorySession(client, "agent-1", "sess_abc")
        session.recall("something", top_k=10)

        import json
        body = json.loads(mock_responses.calls[0].request.body)
        assert body["top_k"] == 10


class TestChatMemorySessionClose:
    def test_close_ends_session(self, client, mock_responses):
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/sessions/sess_abc/end",
            json={"session_id": "sess_abc", "status": "ended"},
            status=200,
        )

        session = ChatMemorySession(client, "agent-1", "sess_abc")
        result = session.close(summary="Test conversation ended")

        assert len(mock_responses.calls) == 1
        import json
        body = json.loads(mock_responses.calls[0].request.body)
        assert body["summary"] == "Test conversation ended"

    def test_close_without_summary(self, client, mock_responses):
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/sessions/sess_xyz/end",
            json={"session_id": "sess_xyz"},
            status=200,
        )

        session = ChatMemorySession(client, "agent-1", "sess_xyz")
        session.close()

        assert len(mock_responses.calls) == 1


class TestChatMemorySessionContextManager:
    def test_context_manager_closes_on_exit(self, client, mock_responses):
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/sessions/sess_cm/end",
            json={"session_id": "sess_cm"},
            status=200,
        )

        session = ChatMemorySession(client, "agent-1", "sess_cm")
        with session:
            pass

        assert len(mock_responses.calls) == 1

    def test_context_manager_closes_on_exception(self, client, mock_responses):
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/sessions/sess_ex/end",
            json={"session_id": "sess_ex"},
            status=200,
        )

        session = ChatMemorySession(client, "agent-1", "sess_ex")
        with pytest.raises(ValueError):
            with session:
                raise ValueError("test error")

        assert len(mock_responses.calls) == 1

    def test_context_manager_swallows_close_error(self, client):
        """close() errors must not propagate out of __exit__."""
        session = ChatMemorySession(client, "agent-1", "sess_err")

        with patch.object(session, "close", side_effect=RuntimeError("network error")):
            with session:
                pass  # should not raise
