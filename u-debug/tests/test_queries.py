"""Tests for the test-queries and banks-periods API endpoints."""

from unittest.mock import MagicMock, patch


def test_get_test_queries_returns_list(client):
    """GET /api/test-queries returns parsed YAML test cases."""
    response = client.get("/api/test-queries")
    assert response.status_code == 200
    data = response.get_json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert "name" in data[0]
    assert "query" in data[0]


def test_get_test_queries_has_combos(client):
    """Each test case includes combos with bank and period."""
    response = client.get("/api/test-queries")
    data = response.get_json()
    for case in data:
        assert "combos" in case
        for combo in case["combos"]:
            assert "bank" in combo
            assert "period" in combo


@patch("debug.api.queries.get_connection")
def test_get_banks_periods(mock_conn, client):
    """GET /api/banks-periods returns bank, period, and source lists."""
    mock_cursor = MagicMock()
    mock_cursor.fetchall.side_effect = [
        [("BMO",), ("RBC",)],
        [("2025_Q4",), ("2026_Q1",)],
        [("investor-slides",), ("pillar3",)],
    ]
    mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
    mock_cursor.__exit__ = MagicMock(return_value=False)

    conn = MagicMock()
    conn.cursor.return_value = mock_cursor
    mock_conn.return_value = conn

    response = client.get("/api/banks-periods")
    assert response.status_code == 200
    data = response.get_json()

    assert data["banks"] == ["BMO", "RBC"]
    assert data["periods"] == ["2025_Q4", "2026_Q1"]
    assert data["sources"] == ["investor-slides", "pillar3"]
    conn.close.assert_called_once()
