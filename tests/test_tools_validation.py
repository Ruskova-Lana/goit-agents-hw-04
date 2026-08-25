"""Тести Pydantic-схем усіх tools: коректні та некоректні вхідні дані."""

import pytest
from pydantic import ValidationError

from tools import HotelCostInput, TransportInput, TripBudgetInput
from hitl import HotelBookingInput
from knowledge import KnowledgeSearchInput


# ================================================================
# TripBudgetInput
# ================================================================

def test_trip_budget_input_valid():
    data = TripBudgetInput(travelers=2, days=5, daily_budget=80)
    assert data.travelers == 2
    assert data.days == 5
    assert data.daily_budget == 80


def test_trip_budget_input_invalid_travelers_zero():
    with pytest.raises(ValidationError):
        TripBudgetInput(travelers=0, days=5, daily_budget=80)


def test_trip_budget_input_invalid_travelers_too_many():
    with pytest.raises(ValidationError):
        TripBudgetInput(travelers=11, days=5, daily_budget=80)


def test_trip_budget_input_invalid_days_range():
    with pytest.raises(ValidationError):
        TripBudgetInput(travelers=2, days=31, daily_budget=80)


def test_trip_budget_input_invalid_negative_budget():
    with pytest.raises(ValidationError):
        TripBudgetInput(travelers=2, days=5, daily_budget=-10)


# ================================================================
# HotelCostInput
# ================================================================

def test_hotel_cost_input_valid():
    data = HotelCostInput(nights=4, price_per_night=100, rooms=1)
    assert data.nights == 4


def test_hotel_cost_input_invalid_nights():
    with pytest.raises(ValidationError):
        HotelCostInput(nights=0, price_per_night=100, rooms=1)


def test_hotel_cost_input_invalid_price():
    with pytest.raises(ValidationError):
        HotelCostInput(nights=4, price_per_night=0, rooms=1)


def test_hotel_cost_input_invalid_rooms_too_many():
    with pytest.raises(ValidationError):
        HotelCostInput(nights=4, price_per_night=100, rooms=6)


def test_hotel_cost_input_default_rooms():
    data = HotelCostInput(nights=4, price_per_night=100)
    assert data.rooms == 1


# ================================================================
# TransportInput
# ================================================================

def test_transport_input_valid():
    data = TransportInput(distance_km=500, travelers=2, priority="fast")
    assert data.priority == "fast"


def test_transport_input_invalid_negative_distance():
    with pytest.raises(ValidationError):
        TransportInput(distance_km=-100, travelers=2, priority="fast")


def test_transport_input_invalid_distance_too_far():
    with pytest.raises(ValidationError):
        TransportInput(distance_km=20001, travelers=2, priority="fast")


def test_transport_input_invalid_priority():
    with pytest.raises(ValidationError):
        TransportInput(distance_km=500, travelers=2, priority="expensive")


def test_transport_input_priority_normalized_case():
    data = TransportInput(distance_km=500, travelers=2, priority="FAST")
    assert data.priority == "fast"


# ================================================================
# HotelBookingInput (ризиковий tool)
# ================================================================

def test_hotel_booking_input_valid():
    data = HotelBookingInput(
        hotel_name="Demo Hotel",
        check_in="2026-09-15",
        nights=4,
        total_cost=400,
    )
    assert data.hotel_name == "Demo Hotel"


def test_hotel_booking_input_invalid_short_name():
    with pytest.raises(ValidationError):
        HotelBookingInput(
            hotel_name="A",
            check_in="2026-09-15",
            nights=4,
            total_cost=400,
        )


def test_hotel_booking_input_invalid_date_format():
    with pytest.raises(ValidationError):
        HotelBookingInput(
            hotel_name="Demo Hotel",
            check_in="15-09-2026",
            nights=4,
            total_cost=400,
        )


def test_hotel_booking_input_invalid_nights_range():
    with pytest.raises(ValidationError):
        HotelBookingInput(
            hotel_name="Demo Hotel",
            check_in="2026-09-15",
            nights=31,
            total_cost=400,
        )


def test_hotel_booking_input_invalid_negative_cost():
    with pytest.raises(ValidationError):
        HotelBookingInput(
            hotel_name="Demo Hotel",
            check_in="2026-09-15",
            nights=4,
            total_cost=-1,
        )


# ================================================================
# KnowledgeSearchInput (agentic RAG)
# ================================================================

def test_knowledge_search_input_valid():
    data = KnowledgeSearchInput(query="travel insurance", top_k=3)
    assert data.top_k == 3


def test_knowledge_search_input_invalid_short_query():
    with pytest.raises(ValidationError):
        KnowledgeSearchInput(query="hi", top_k=3)


def test_knowledge_search_input_invalid_top_k_too_large():
    with pytest.raises(ValidationError):
        KnowledgeSearchInput(query="travel insurance", top_k=10)


def test_knowledge_search_input_default_top_k():
    data = KnowledgeSearchInput(query="travel insurance")
    assert data.top_k == 3
