"""Tests for RecalledMemory.score priority: smart_score > weighted_score > raw score."""

from dakera.models import RecalledMemory, RecallResponse


def test_smart_score_takes_priority():
    data = {
        "id": "m1", "content": "test", "memory_type": "episodic", "importance": 0.8,
        "score": 0.5, "weighted_score": 0.7, "smart_score": 0.9,
    }
    m = RecalledMemory.from_dict(data)
    assert m.score == pytest.approx(0.9), ".score must equal smart_score when present"
    assert m.smart_score == pytest.approx(0.9)
    assert m.weighted_score == pytest.approx(0.7)


def test_weighted_score_fallback():
    data = {
        "id": "m2", "content": "test", "memory_type": "episodic", "importance": 0.8,
        "score": 0.5, "weighted_score": 0.7,
    }
    m = RecalledMemory.from_dict(data)
    assert m.score == pytest.approx(0.7), ".score must equal weighted_score when smart_score absent"
    assert m.smart_score is None
    assert m.weighted_score == pytest.approx(0.7)


def test_raw_score_fallback():
    data = {
        "id": "m3", "content": "test", "memory_type": "episodic", "importance": 0.8,
        "score": 0.55,
    }
    m = RecalledMemory.from_dict(data)
    assert m.score == pytest.approx(0.55), ".score must equal raw score when neither smart_score nor weighted_score"
    assert m.smart_score is None
    assert m.weighted_score is None


def test_recall_response_normalize_forwards_smart_score():
    raw = {
        "memories": [
            {
                "memory": {"id": "m4", "content": "nested", "memory_type": "episodic", "importance": 0.9,
                           "created_at": "2026-01-01T00:00:00Z", "tags": []},
                "score": 0.4,
                "weighted_score": 0.6,
                "smart_score": 0.85,
            }
        ]
    }
    resp = RecallResponse.from_dict(raw)
    assert len(resp.memories) == 1
    m = resp.memories[0]
    assert m.score == pytest.approx(0.85)
    assert m.smart_score == pytest.approx(0.85)
    assert m.weighted_score == pytest.approx(0.6)


def test_recall_response_normalize_without_smart_score():
    raw = {
        "memories": [
            {
                "memory": {"id": "m5", "content": "nested2", "memory_type": "semantic", "importance": 0.7,
                           "created_at": "2026-01-01T00:00:00Z", "tags": []},
                "score": 0.3,
                "weighted_score": 0.55,
            }
        ]
    }
    resp = RecallResponse.from_dict(raw)
    m = resp.memories[0]
    assert m.score == pytest.approx(0.55)
    assert m.smart_score is None


import pytest
