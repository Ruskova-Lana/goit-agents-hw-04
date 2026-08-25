"""Тести виконання tools через .invoke() — коректний JSON-результат + обробка помилок.

Кожен tool повертає JSON-рядок стандартної структури:
    {"status": "success", "data": {...}}
    {"status": "error", "error": "..."}
"""

import json

import pytest

from tools import calculate_trip_budget, estimate_hotel_cost, recommend_transport
from hitl import book_hotel
from knowledge import search_knowledge
from tool_utils import error_json, safe_tool_invoke, success_json


def test_calculate_trip_budget_result():
    result = json.loads(
        calculate_trip_budget.invoke({"travelers": 2, "days": 5, "daily_budget": 80})
    )
    assert result["status"] == "success"
    assert result["data"]["total_budget"] == 800.0


def test_calculate_trip_budget_invalid_args_raise():
    with pytest.raises(Exception):
        calculate_trip_budget.invoke({"travelers": 0, "days": 5, "daily_budget": 80})


def test_estimate_hotel_cost_result():
    result = json.loads(
        estimate_hotel_cost.invoke({"nights": 4, "price_per_night": 100, "rooms": 1})
    )
    assert result["status"] == "success"
    assert result["data"]["total_cost"] == 400.0


def test_recommend_transport_short_distance():
    result = json.loads(
        recommend_transport.invoke(
            {"distance_km": 5, "travelers": 1, "priority": "balanced"}
        )
    )
    assert result["status"] == "success"
    transport = result["data"]["recommended_transport"]
    assert "таксі" in transport or "громадський" in transport


def test_recommend_transport_long_distance_uses_plane():
    result = json.loads(
        recommend_transport.invoke(
            {"distance_km": 1500, "travelers": 2, "priority": "fast"}
        )
    )
    assert result["status"] == "success"
    assert "літак" in result["data"]["recommended_transport"]


def test_book_hotel_result_contains_booking_id():
    result = json.loads(
        book_hotel.invoke(
            {
                "hotel_name": "Demo Travel Hotel",
                "check_in": "2026-09-15",
                "nights": 4,
                "total_cost": 400,
            }
        )
    )
    assert result["status"] == "success"
    assert result["data"]["booking_id"] == "DEMO-BOOKING-001"


def test_book_hotel_invalid_args_raise():
    with pytest.raises(Exception):
        book_hotel.invoke(
            {
                "hotel_name": "A",
                "check_in": "2026-09-15",
                "nights": 4,
                "total_cost": 400,
            }
        )


def test_search_knowledge_finds_relevant_document():
    result = json.loads(
        search_knowledge.invoke(
            {"query": "туристичне страхування для подорожі", "top_k": 2}
        )
    )
    assert result["status"] == "success"
    titles = [item["title"] for item in result["data"]["results"]]
    assert any("insurance" in title.lower() for title in titles)


def test_search_knowledge_invalid_short_query_raises():
    with pytest.raises(Exception):
        search_knowledge.invoke({"query": "hi", "top_k": 2})


# ================================================================
# tool_utils: success_json / error_json / safe_tool_invoke
# ================================================================

def test_success_json_structure():
    parsed = json.loads(success_json({"a": 1}))
    assert parsed == {"status": "success", "data": {"a": 1}}


def test_error_json_structure():
    parsed = json.loads(error_json("щось пішло не так"))
    assert parsed == {"status": "error", "error": "щось пішло не так"}


def test_safe_tool_invoke_returns_json_on_success():
    result = json.loads(
        safe_tool_invoke(
            calculate_trip_budget,
            {"travelers": 2, "days": 5, "daily_budget": 80},
        )
    )
    assert result["status"] == "success"


def test_safe_tool_invoke_never_raises_on_invalid_args():
    # На відміну від .invoke() напряму, safe_tool_invoke() не піднімає
    # виняток навіть при невалідних аргументах — гарантує JSON-контракт.
    result = json.loads(
        safe_tool_invoke(calculate_trip_budget, {"travelers": 0, "days": 5, "daily_budget": 80})
    )
    assert result["status"] == "error"
    assert "error" in result
