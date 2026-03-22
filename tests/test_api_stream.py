"""Tests for POST /predict/stream API endpoint."""
import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def _make_transaction(**overrides) -> dict:
    """Build a minimal valid transaction payload."""
    tx = {
        "amount": 100.0,
        "time": 0.0,
        **{f"v{i}": 0.0 for i in range(1, 29)},
    }
    tx.update(overrides)
    return tx


def _make_batch(n: int = 3) -> dict:
    return {"transactions": [_make_transaction(amount=float(i + 1) * 10) for i in range(n)]}


class TestStreamPredict:
    def test_stream_endpoint(self):
        """POST /predict/stream returns 200 with results and summary."""
        payload = _make_batch(5)
        resp = client.post("/predict/stream", json=payload)
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert "results" in data
        assert "summary" in data
        assert "cost_estimate" in data

    def test_results_count_matches_input(self):
        """Number of results equals number of submitted transactions."""
        n = 7
        resp = client.post("/predict/stream", json=_make_batch(n))
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == n
        assert data["summary"]["total_processed"] == n

    def test_result_fields(self):
        """Each result contains required per-transaction fields."""
        resp = client.post("/predict/stream", json=_make_batch(2))
        data = resp.json()
        required = {"index", "fraud_probability", "decision", "confidence", "latency_ms"}
        for r in data["results"]:
            assert required.issubset(r.keys()), f"Missing keys in {r}"

    def test_summary_fields(self):
        """Summary contains all expected aggregate keys."""
        resp = client.post("/predict/stream", json=_make_batch(4))
        summary = resp.json()["summary"]
        expected = {
            "total_processed", "fraud_detected", "approve_count",
            "review_count", "block_count", "avg_fraud_probability",
            "avg_processing_time_ms", "total_latency_ms",
        }
        assert expected.issubset(summary.keys())

    def test_empty_batch(self):
        """Empty transaction list should be rejected (min_length=1)."""
        resp = client.post("/predict/stream", json={"transactions": []})
        assert resp.status_code == 422  # Pydantic validation error

    def test_cost_estimate(self):
        """cost_estimate block is present with non-negative total."""
        resp = client.post("/predict/stream", json=_make_batch(10))
        cost = resp.json()["cost_estimate"]
        assert "total_cost_usd" in cost
        assert "cost_per_transaction_usd" in cost
        assert cost["total_cost_usd"] >= 0.0

    def test_fraud_detected_count_consistency(self):
        """block_count in summary equals fraud_detected count."""
        resp = client.post("/predict/stream", json=_make_batch(6))
        summary = resp.json()["summary"]
        assert summary["block_count"] == summary["fraud_detected"]

    def test_decision_counts_sum_to_total(self):
        """approve + review + block should equal total_processed."""
        resp = client.post("/predict/stream", json=_make_batch(8))
        s = resp.json()["summary"]
        assert s["approve_count"] + s["review_count"] + s["block_count"] == s["total_processed"]
