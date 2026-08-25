"""Спільні хелпери для JSON-виходу tools та безпечного виклику tools.

Кожен domain tool повертає JSON-рядок зі стандартною структурою:
    {"status": "success", "data": {...}}
    {"status": "error", "error": "..."}

safe_tool_invoke() додатково перехоплює помилки, які виникають ДО тіла
tool-функції (наприклад Pydantic-валідація args_schema під час .invoke()),
щоб один невдалий виклик tool ніколи не "впав" увесь LangGraph граф.
"""

import json


def success_json(data: dict) -> str:
    """Формує JSON-рядок успішного результату tool."""

    return json.dumps(
        {"status": "success", "data": data},
        ensure_ascii=False,
    )


def error_json(message: str) -> str:
    """Формує JSON-рядок помилки tool."""

    return json.dumps(
        {"status": "error", "error": message},
        ensure_ascii=False,
    )


def safe_tool_invoke(tool_function, args: dict) -> str:
    """Викликає tool.invoke(args) і гарантує JSON-рядок у відповідь.

    Перехоплює будь-які винятки (включно з Pydantic ValidationError,
    що виникає під час валідації args_schema ще до виконання тіла
    функції) і повертає їх у вигляді {"status": "error", "error": ...},
    а не дає їм зупинити виконання графа.
    """

    try:
        return tool_function.invoke(args)
    except Exception as exc:
        return error_json(f"Помилка виконання tool {tool_function.name}: {exc}")
