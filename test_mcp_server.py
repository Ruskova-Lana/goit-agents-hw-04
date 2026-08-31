"""Async unit-тести для mcp_server.py.

Перевіряють усі три типи MCP-примітивів через офіційний API FastMCP:
mcp.list_tools() / mcp.call_tool() / mcp.list_resources() /
mcp.read_resource() / mcp.list_prompts() / mcp.get_prompt().

Запуск:
    python -m pytest test_mcp_server.py -v
"""

import json

import pytest

from mcp_server import mcp


def _tool_result_json(call_result) -> dict:
    """call_tool() повертає (content_blocks, structured_result).

    Тіло кожного tool повертає JSON-рядок (success_json/error_json), тому
    парсимо text останнього TextContent-блока.
    """

    content_blocks, _structured = call_result
    return json.loads(content_blocks[0].text)


# ================================================================
# TOOLS — list_tools()
# ================================================================

@pytest.mark.asyncio
async def test_list_tools_returns_all_six_tools():
    tools = await mcp.list_tools()
    names = {t.name for t in tools}

    assert names == {
        "calculate_trip_budget",
        "estimate_hotel_cost",
        "recommend_transport",
        "search_travel_knowledge",
        "convert_currency",
        "book_hotel",
    }


@pytest.mark.asyncio
async def test_tools_have_non_empty_descriptions_for_llm():
    """LLM читає docstrings через description — вони не повинні бути пустими."""

    tools = await mcp.list_tools()
    for tool in tools:
        assert tool.description and len(tool.description.strip()) > 20
        assert tool.inputSchema is not None


# ================================================================
# TOOLS — call_tool() happy path
# ================================================================

@pytest.mark.asyncio
async def test_calculate_trip_budget_success():
    result = await mcp.call_tool(
        "calculate_trip_budget", {"travelers": 2, "days": 5, "daily_budget": 80}
    )
    payload = _tool_result_json(result)

    assert payload["status"] == "success"
    assert payload["data"]["total_budget"] == 800.0
    assert payload["data"]["currency"] == "EUR"


@pytest.mark.asyncio
async def test_estimate_hotel_cost_success():
    result = await mcp.call_tool(
        "estimate_hotel_cost", {"nights": 4, "price_per_night": 100, "rooms": 1}
    )
    payload = _tool_result_json(result)

    assert payload["status"] == "success"
    assert payload["data"]["total_cost"] == 400.0


@pytest.mark.asyncio
async def test_recommend_transport_success():
    result = await mcp.call_tool(
        "recommend_transport", {"distance_km": 1200, "travelers": 2, "priority": "fast"}
    )
    payload = _tool_result_json(result)

    assert payload["status"] == "success"
    assert payload["data"]["recommended_transport"] == "літак"


@pytest.mark.asyncio
async def test_search_travel_knowledge_success():
    result = await mcp.call_tool(
        "search_travel_knowledge", {"query": "travel insurance", "top_k": 2}
    )
    payload = _tool_result_json(result)

    assert payload["status"] == "success"
    assert len(payload["data"]["results"]) <= 2
    assert all("title" in item and "text" in item for item in payload["data"]["results"])


@pytest.mark.asyncio
async def test_book_hotel_success():
    result = await mcp.call_tool(
        "book_hotel",
        {"hotel_name": "Demo Travel Hotel", "check_in": "2026-09-15", "nights": 4, "total_cost": 400},
    )
    payload = _tool_result_json(result)

    assert payload["status"] == "success"
    assert payload["data"]["booking_id"] == "DEMO-BOOKING-001"


@pytest.mark.asyncio
async def test_book_hotel_invalid_date_returns_error_json():
    result = await mcp.call_tool(
        "book_hotel",
        {"hotel_name": "Demo Travel Hotel", "check_in": "15-09-2026", "nights": 4, "total_cost": 400},
    )
    payload = _tool_result_json(result)

    assert payload["status"] == "error"


@pytest.mark.asyncio
async def test_convert_currency_success():
    result = await mcp.call_tool(
        "convert_currency", {"amount": 100, "from_currency": "eur", "to_currency": "usd"}
    )
    payload = _tool_result_json(result)

    assert payload["status"] == "success"
    assert payload["data"]["converted_amount"] == 108.0
    assert payload["data"]["to_currency"] == "USD"


# ================================================================
# TOOLS — call_tool() error path (Pydantic/manual validation)
# ================================================================

@pytest.mark.asyncio
async def test_calculate_trip_budget_invalid_travelers_returns_error_json():
    result = await mcp.call_tool(
        "calculate_trip_budget", {"travelers": 0, "days": 5, "daily_budget": 80}
    )
    payload = _tool_result_json(result)

    assert payload["status"] == "error"
    assert "мандрівник" in payload["error"]


@pytest.mark.asyncio
async def test_convert_currency_unknown_currency_returns_error_json():
    result = await mcp.call_tool(
        "convert_currency", {"amount": 50, "from_currency": "EUR", "to_currency": "XYZ"}
    )
    payload = _tool_result_json(result)

    assert payload["status"] == "error"
    assert "XYZ" in payload["error"]


# ================================================================
# RESOURCES — list_resources() / read_resource()
# ================================================================

@pytest.mark.asyncio
async def test_list_resources_returns_both_resources():
    resources = await mcp.list_resources()
    uris = {str(r.uri) for r in resources}

    assert uris == {"travel://knowledge-base", "travel://currency-rates"}


@pytest.mark.asyncio
async def test_read_resource_currency_rates_contains_eur_base():
    contents = list(await mcp.read_resource("travel://currency-rates"))

    assert len(contents) == 1
    payload = json.loads(contents[0].content)

    assert payload["base_currency"] == "EUR"
    assert payload["rates"]["EUR"] == 1.0
    assert "USD" in payload["rates"]


@pytest.mark.asyncio
async def test_read_resource_knowledge_base_lists_documents():
    contents = list(await mcp.read_resource("travel://knowledge-base"))
    payload = json.loads(contents[0].content)

    assert len(payload["documents"]) == 10
    assert all("id" in doc and "title" in doc for doc in payload["documents"])


# ================================================================
# PROMPTS — list_prompts() / get_prompt()
# ================================================================

@pytest.mark.asyncio
async def test_list_prompts_returns_both_prompts():
    prompts = await mcp.list_prompts()
    names = {p.name for p in prompts}

    assert names == {"trip_planning", "budget_summary"}


@pytest.mark.asyncio
async def test_get_prompt_trip_planning_fills_arguments():
    result = await mcp.get_prompt(
        "trip_planning",
        {"destination": "Lisbon", "days": "5", "travelers": "2", "priority": "fast"},
    )

    text = result.messages[0].content.text
    assert "Lisbon" in text
    assert "5" in text
    assert "recommend_transport" in text


@pytest.mark.asyncio
async def test_get_prompt_budget_summary_fills_arguments():
    result = await mcp.get_prompt(
        "budget_summary", {"trip_budget": "800", "hotel_cost": "400", "currency": "EUR"}
    )

    text = result.messages[0].content.text
    assert "800" in text
    assert "400" in text
    assert "convert_currency" in text
