"""Tests for T-I-F reliability scoring (T-I-F RFC Phase 3)."""

import pytest

from dakera.models import (
    FeedbackHistoryEntry,
    FeedbackHistoryResponse,
    FeedbackSignal,
    TifScore,
)


def _make_history(*signals: str) -> FeedbackHistoryResponse:
    entries = [
        FeedbackHistoryEntry(
            signal=FeedbackSignal(s),
            timestamp=0,
            old_importance=0.5,
            new_importance=0.5,
        )
        for s in signals
    ]
    return FeedbackHistoryResponse(memory_id="test-mem", entries=entries)


class TestTifScoreFromFeedbackHistory:
    def test_no_feedback_returns_max_indeterminacy(self):
        score = TifScore.from_feedback_history(_make_history())
        assert score.truth == 0.0
        assert score.indeterminacy == 1.0
        assert score.falsity == 0.0
        assert score.feedback_count == 0

    def test_all_upvotes(self):
        score = TifScore.from_feedback_history(_make_history("upvote", "upvote", "upvote"))
        assert score.truth == pytest.approx(1.0)
        assert score.falsity == pytest.approx(0.0)
        assert score.indeterminacy == pytest.approx(0.0)
        assert score.feedback_count == 3

    def test_all_downvotes(self):
        score = TifScore.from_feedback_history(_make_history("downvote", "downvote"))
        assert score.truth == pytest.approx(0.0)
        assert score.falsity == pytest.approx(0.8)
        assert score.indeterminacy == pytest.approx(0.2)
        assert score.feedback_count == 2

    def test_all_flags(self):
        score = TifScore.from_feedback_history(_make_history("flag", "flag"))
        assert score.truth == pytest.approx(0.0)
        assert score.indeterminacy == pytest.approx(1.0)
        assert score.falsity == pytest.approx(0.0)
        assert score.feedback_count == 2

    def test_mixed_signals(self):
        # 4 upvotes, 2 downvotes, 4 flags → total 10
        score = TifScore.from_feedback_history(
            _make_history(
                "upvote", "upvote", "upvote", "upvote",
                "downvote", "downvote",
                "flag", "flag", "flag", "flag",
            )
        )
        assert score.truth == pytest.approx(0.4)
        assert score.falsity == pytest.approx(0.2)
        assert score.indeterminacy == pytest.approx(0.4)
        assert score.feedback_count == 10

    def test_positive_alias_counts_as_upvote(self):
        score = TifScore.from_feedback_history(_make_history("positive", "positive", "downvote"))
        assert score.truth == pytest.approx(2 / 3)
        assert score.falsity == pytest.approx(1 / 3)

    def test_negative_alias_counts_as_downvote(self):
        score = TifScore.from_feedback_history(_make_history("upvote", "negative", "negative"))
        assert score.falsity == pytest.approx(2 / 3)
        assert score.truth == pytest.approx(1 / 3)

    def test_proportions_sum_to_one(self):
        score = TifScore.from_feedback_history(_make_history("upvote", "downvote", "flag"))
        assert score.truth + score.indeterminacy + score.falsity == pytest.approx(1.0)


class TestTifScoreClassification:
    def test_confident_reuse(self):
        score = TifScore(truth=0.75, indeterminacy=0.1, falsity=0.15, feedback_count=10)
        assert score.classification == "confident_reuse"

    def test_surface_contradiction(self):
        score = TifScore(truth=0.1, indeterminacy=0.1, falsity=0.80, feedback_count=10)
        assert score.classification == "surface_contradiction"

    def test_ask_clarification(self):
        score = TifScore(truth=0.1, indeterminacy=0.75, falsity=0.15, feedback_count=10)
        assert score.classification == "ask_clarification"

    def test_verify_before_use(self):
        # truth=0.40, indeterminacy=0.30, falsity=0.30 — no threshold met
        score = TifScore(truth=0.40, indeterminacy=0.30, falsity=0.30, feedback_count=10)
        assert score.classification == "verify_before_use"

    def test_verify_before_use_from_history(self):
        # 4 upvotes, 3 downvotes, 3 flags → truth=0.4, falsity=0.3, indeterminacy=0.3
        score = TifScore.from_feedback_history(
            _make_history(
                "upvote", "upvote", "upvote", "upvote",
                "downvote", "downvote", "downvote",
                "flag", "flag", "flag",
            )
        )
        assert score.classification == "verify_before_use"

    def test_falsity_threshold_boundary(self):
        score = TifScore(truth=0.0, indeterminacy=0.0, falsity=0.50, feedback_count=2)
        assert score.classification == "surface_contradiction"

    def test_indeterminacy_threshold_boundary(self):
        score = TifScore(truth=0.0, indeterminacy=0.50, falsity=0.0, feedback_count=2)
        assert score.classification == "ask_clarification"

    def test_truth_threshold_boundary(self):
        score = TifScore(truth=0.70, indeterminacy=0.20, falsity=0.10, feedback_count=10)
        assert score.classification == "confident_reuse"

    def test_no_feedback_classification(self):
        score = TifScore.from_feedback_history(_make_history())
        assert score.classification == "ask_clarification"

    def test_falsity_takes_priority_over_indeterminacy(self):
        # Both falsity >= 0.5 and indeterminacy >= 0.5 — falsity checked first
        score = TifScore(truth=0.0, indeterminacy=0.5, falsity=0.5, feedback_count=2)
        assert score.classification == "surface_contradiction"


class TestTifScoreFromMetadata:
    def test_round_trip(self):
        original = TifScore(truth=0.75, indeterminacy=0.15, falsity=0.10, feedback_count=20)
        data = {
            "truth": original.truth,
            "indeterminacy": original.indeterminacy,
            "falsity": original.falsity,
            "feedback_count": original.feedback_count,
        }
        restored = TifScore.from_metadata(data)
        assert restored.truth == pytest.approx(original.truth)
        assert restored.indeterminacy == pytest.approx(original.indeterminacy)
        assert restored.falsity == pytest.approx(original.falsity)
        assert restored.feedback_count == original.feedback_count

    def test_missing_feedback_count_defaults_to_zero(self):
        score = TifScore.from_metadata({"truth": 0.8, "indeterminacy": 0.1, "falsity": 0.1})
        assert score.feedback_count == 0


class TestThinEvidence:
    def test_single_upvote_is_not_confident(self):
        score = TifScore.from_feedback_history(_make_history("upvote"))
        assert score.truth + score.indeterminacy + score.falsity == pytest.approx(1.0)
        assert score.indeterminacy > 0.0
        assert score.truth < 0.70
        assert score.classification == "verify_before_use"

    def test_two_upvotes_reach_confident(self):
        score = TifScore.from_feedback_history(_make_history("upvote", "upvote"))
        assert score.truth == pytest.approx(0.8)
        assert score.indeterminacy == pytest.approx(0.2)
        assert score.classification == "confident_reuse"

    def test_three_upvotes_no_base_indeterminacy(self):
        score = TifScore.from_feedback_history(_make_history("upvote", "upvote", "upvote"))
        assert score.truth == pytest.approx(1.0)
        assert score.indeterminacy == pytest.approx(0.0)
        assert score.classification == "confident_reuse"


class TestGoldenVectors:
    """Canonical T-I-F v1 contract — identical across MCP + Python + JS + Rust + Go."""

    def test_no_feedback(self):
        s = TifScore.from_feedback_history(_make_history())
        assert s.truth == pytest.approx(0.0)
        assert s.indeterminacy == pytest.approx(1.0)
        assert s.falsity == pytest.approx(0.0)
        assert s.classification == "ask_clarification"

    def test_one_upvote(self):
        s = TifScore.from_feedback_history(_make_history("upvote"))
        assert s.truth == pytest.approx(2 / 3, abs=1e-4)
        assert s.indeterminacy == pytest.approx(1 / 3, abs=1e-4)
        assert s.falsity == pytest.approx(0.0)
        assert s.classification == "verify_before_use"

    def test_two_upvotes(self):
        s = TifScore.from_feedback_history(_make_history("upvote", "upvote"))
        assert s.truth == pytest.approx(0.8)
        assert s.indeterminacy == pytest.approx(0.2)
        assert s.falsity == pytest.approx(0.0)
        assert s.classification == "confident_reuse"

    def test_three_upvotes(self):
        s = TifScore.from_feedback_history(_make_history("upvote", "upvote", "upvote"))
        assert s.truth == pytest.approx(1.0)
        assert s.indeterminacy == pytest.approx(0.0)
        assert s.falsity == pytest.approx(0.0)
        assert s.classification == "confident_reuse"

    def test_two_downvotes(self):
        s = TifScore.from_feedback_history(_make_history("downvote", "downvote"))
        assert s.truth == pytest.approx(0.0)
        assert s.indeterminacy == pytest.approx(0.2)
        assert s.falsity == pytest.approx(0.8)
        assert s.classification == "surface_contradiction"

    def test_two_flags(self):
        s = TifScore.from_feedback_history(_make_history("flag", "flag"))
        assert s.truth == pytest.approx(0.0)
        assert s.indeterminacy == pytest.approx(1.0)
        assert s.falsity == pytest.approx(0.0)
        assert s.classification == "ask_clarification"

    def test_8up_1down_1flag(self):
        s = TifScore.from_feedback_history(
            _make_history(
                "upvote", "upvote", "upvote", "upvote",
                "upvote", "upvote", "upvote", "upvote",
                "downvote", "flag",
            )
        )
        assert s.truth == pytest.approx(0.8)
        assert s.indeterminacy == pytest.approx(0.1)
        assert s.falsity == pytest.approx(0.1)
        assert s.classification == "confident_reuse"

    def test_3down_3flag(self):
        s = TifScore.from_feedback_history(
            _make_history("downvote", "downvote", "downvote", "flag", "flag", "flag")
        )
        assert s.truth == pytest.approx(0.0)
        assert s.indeterminacy == pytest.approx(0.5)
        assert s.falsity == pytest.approx(0.5)
        assert s.classification == "surface_contradiction"


class TestEvaluateTif:
    """Integration tests for DakeraClient.evaluate_tif()."""

    def test_evaluate_tif_happy_path(self):
        """evaluate_tif fetches feedback history and returns a TifScore."""
        import responses as rsps_lib
        from dakera import DakeraClient

        client = DakeraClient("http://localhost:3000")
        with rsps_lib.RequestsMock() as rsps:
            rsps.add(
                rsps_lib.GET,
                "http://localhost:3000/v1/memories/mem-abc/feedback",
                json={
                    "memory_id": "mem-abc",
                    "entries": [
                        {"signal": "upvote", "timestamp": 1, "old_importance": 0.5, "new_importance": 0.6},
                        {"signal": "upvote", "timestamp": 2, "old_importance": 0.6, "new_importance": 0.7},
                    ],
                },
                status=200,
            )
            score = client.evaluate_tif("mem-abc")
            assert len(rsps.calls) == 1

        assert isinstance(score, TifScore)
        # 2 upvotes: TIF smoothing yields truth=0.8, not 1.0
        assert score.truth == pytest.approx(0.8)
        assert score.falsity == pytest.approx(0.0)
        assert score.feedback_count == 2

    def test_evaluate_tif_no_feedback_returns_indeterminate(self):
        """evaluate_tif with no feedback signals returns indeterminacy=1.0."""
        import responses as rsps_lib
        from dakera import DakeraClient

        client = DakeraClient("http://localhost:3000")
        with rsps_lib.RequestsMock() as rsps:
            rsps.add(
                rsps_lib.GET,
                "http://localhost:3000/v1/memories/mem-xyz/feedback",
                json={"memory_id": "mem-xyz", "entries": []},
                status=200,
            )
            score = client.evaluate_tif("mem-xyz")

        assert score.indeterminacy == pytest.approx(1.0)
        assert score.truth == pytest.approx(0.0)
        assert score.falsity == pytest.approx(0.0)

    def test_evaluate_tif_server_error_propagates(self):
        """evaluate_tif propagates DakeraError from the feedback history fetch."""
        import responses as rsps_lib
        from dakera import DakeraClient
        from dakera.exceptions import DakeraError

        client = DakeraClient("http://localhost:3000")
        with rsps_lib.RequestsMock() as rsps:
            rsps.add(
                rsps_lib.GET,
                "http://localhost:3000/v1/memories/mem-bad/feedback",
                json={"error": "internal server error"},
                status=500,
            )
            with pytest.raises(DakeraError):
                client.evaluate_tif("mem-bad")
