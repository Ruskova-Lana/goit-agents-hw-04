"""MCP-сервер туристичного асистента (FastMCP, офіційний MCP Python SDK).

Продовження ДЗ1/ДЗ2/ДЗ4-Завд.1 — той самий домен (подорожі), той самий
JSON-контракт результатів (`tool_utils.success_json`/`error_json`), але
подано через стандартизований протокол MCP замість LangChain `@tool`:

- **Tools** (6)   — виконуючі функції: 3 перевикористані з `tools.py`
  (`calculate_trip_budget`, `estimate_hotel_cost`, `recommend_transport`),
  1 перевикористаний Agentic RAG tool з `knowledge.py`
  (`search_travel_knowledge`), 1 новий (`convert_currency`), 1 ризиковий
  tool з `hitl.py` (`book_hotel`) — потребує Human-in-the-Loop approval
  (ДЗ4 Завд. 4, `guardrails.requires_human_approval`); HITL-обгортку
  `interrupt()` над ним демонструє `mas_langgraph.py` (billing-агент).
- **Resources** (2) — read-only довідники: список документів бази знань
  та таблиця курсів валют.
- **Prompts** (2)   — шаблони: план подорожі та підсумковий звіт по бюджету.

Запуск як окремий MCP-сервер (stdio):
    python mcp_server.py
"""

from __future__ import annotations

import sys
from typing import Literal

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from mcp.server.fastmcp import FastMCP

from knowledge import KNOWLEDGE_DOCUMENTS, collection
from tool_utils import error_json, success_json


mcp = FastMCP(
    "travel-assistant",
    instructions=(
        "MCP-сервер туристичного асистента: розрахунок бюджету подорожі, "
        "вартості готелю, рекомендація транспорту, конвертація валют та "
        "пошук у внутрішній базі знань про подорожі."
    ),
)


# ================================================================
# CURRENCY REFERENCE DATA (для convert_currency tool + резервного resource)
# ================================================================

# Фіксовані курси відносно EUR (демо-дані, не live rates).
CURRENCY_RATES_EUR = {
    "EUR": 1.0,
    "USD": 1.08,
    "GBP": 0.86,
    "UAH": 45.20,
    "PLN": 4.30,
}


# ================================================================
# TOOLS (5)
# ================================================================

@mcp.tool()
def calculate_trip_budget(travelers: int, days: int, daily_budget: float) -> str:
    """Розрахувати орієнтовний загальний бюджет подорожі.

    Використовуй цей інструмент, коли користувач хоче дізнатися, скільки
    приблизно коштуватиме поїздка з урахуванням кількості мандрівників,
    тривалості подорожі та щоденного бюджету на одну людину.

    Args:
        travelers: Кількість мандрівників. Допустиме значення: від 1 до 10.
        days: Тривалість подорожі у днях. Допустиме значення: від 1 до 30.
        daily_budget: Щоденний бюджет на одну людину в євро (> 0).

    Returns:
        JSON-рядок {"status": "success", "data": {"total_budget": ...,
        "currency": "EUR", ...}} або {"status": "error", "error": "..."}
        у разі некоректних вхідних даних.
    """

    try:
        if not 1 <= travelers <= 10:
            raise ValueError("Кількість мандрівників повинна бути від 1 до 10.")
        if not 1 <= days <= 30:
            raise ValueError("Кількість днів повинна бути від 1 до 30.")
        if daily_budget <= 0:
            raise ValueError("Щоденний бюджет повинен бути більшим за 0.")

        total = travelers * days * daily_budget

        return success_json(
            {
                "total_budget": round(total, 2),
                "currency": "EUR",
                "travelers": travelers,
                "days": days,
                "daily_budget": daily_budget,
            }
        )
    except Exception as exc:
        return error_json(f"Не вдалося розрахувати бюджет: {exc}")


@mcp.tool()
def estimate_hotel_cost(nights: int, price_per_night: float, rooms: int = 1) -> str:
    """Розрахувати загальну вартість проживання в готелі.

    Використовуй цей інструмент, коли користувач хоче оцінити витрати на
    готель за відомою кількістю ночей, номерів та ціною одного номера за
    ніч.

    Args:
        nights: Кількість ночей у готелі. Допустиме значення: від 1 до 30.
        price_per_night: Вартість одного номера за одну ніч у євро (> 0).
        rooms: Кількість номерів. Допустиме значення: від 1 до 5.

    Returns:
        JSON-рядок {"status": "success", "data": {"total_cost": ...,
        "currency": "EUR", ...}} або {"status": "error", "error": "..."}
        у разі некоректних вхідних даних.
    """

    try:
        if not 1 <= nights <= 30:
            raise ValueError("Кількість ночей повинна бути від 1 до 30.")
        if price_per_night <= 0:
            raise ValueError("Вартість ночі повинна бути більшою за 0.")
        if not 1 <= rooms <= 5:
            raise ValueError("Кількість номерів повинна бути від 1 до 5.")

        total = nights * price_per_night * rooms

        return success_json(
            {
                "total_cost": round(total, 2),
                "currency": "EUR",
                "nights": nights,
                "price_per_night": price_per_night,
                "rooms": rooms,
            }
        )
    except Exception as exc:
        return error_json(f"Не вдалося розрахувати вартість готелю: {exc}")


@mcp.tool()
def recommend_transport(
    distance_km: float,
    travelers: int = 1,
    priority: Literal["cheap", "fast", "balanced"] = "balanced",
) -> str:
    """Порекомендувати вид транспорту для подорожі.

    Використовуй цей інструмент, коли користувач хоче отримати
    рекомендацію щодо транспорту залежно від відстані та пріоритету:
    дешевше (cheap), швидше (fast) або збалансований варіант (balanced).

    Args:
        distance_km: Відстань між пунктами подорожі у кілометрах
            (> 0 та <= 20000).
        travelers: Кількість мандрівників. Допустиме значення: від 1 до 10.
        priority: Пріоритет подорожі — "cheap", "fast" або "balanced".

    Returns:
        JSON-рядок {"status": "success", "data": {"recommended_transport":
        ..., ...}} або {"status": "error", "error": "..."} у разі
        некоректних вхідних даних.
    """

    try:
        if not 0 < distance_km <= 20000:
            raise ValueError("Відстань повинна бути в діапазоні (0, 20000] км.")
        if not 1 <= travelers <= 10:
            raise ValueError("Кількість мандрівників повинна бути від 1 до 10.")

        if distance_km < 10:
            transport = "громадський транспорт або таксі"
        elif distance_km < 300:
            transport = {
                "cheap": "автобус",
                "fast": "потяг",
                "balanced": "потяг або автобус",
            }[priority]
        elif distance_km < 1000:
            transport = {
                "cheap": "автобус або потяг",
                "fast": "літак",
                "balanced": "потяг",
            }[priority]
        else:
            transport = "літак"

        return success_json(
            {
                "recommended_transport": transport,
                "distance_km": distance_km,
                "travelers": travelers,
                "priority": priority,
            }
        )
    except Exception as exc:
        return error_json(f"Не вдалося порекомендувати транспорт: {exc}")


@mcp.tool()
def search_travel_knowledge(query: str, top_k: int = 3) -> str:
    """Знайти довідкову інформацію у внутрішній travel knowledge base (RAG).

    Використовуй цей інструмент, коли для відповіді потрібні довідкові
    знання про: документи, візові вимоги, страхування, багаж, правила
    подорожей, готельні правила, підготовку до міжнародної поїздки.
    Не використовуй цей інструмент для арифметичних розрахунків — для
    цього є calculate_trip_budget / estimate_hotel_cost.

    Args:
        query: Пошуковий запит до бази знань (мінімум 3 символи).
        top_k: Кількість найбільш релевантних документів (від 1 до 5).

    Returns:
        JSON-рядок {"status": "success", "data": {"results": [{"title":
        ..., "text": ...}, ...]}} або {"status": "error", "error": "..."}
        у разі некоректних вхідних даних.
    """

    try:
        query = query.strip()
        if len(query) < 3:
            raise ValueError("Пошуковий запит повинен містити щонайменше 3 символи.")
        if not 1 <= top_k <= 5:
            raise ValueError("top_k повинен бути від 1 до 5.")

        results = collection.query(query_texts=[query], n_results=top_k)

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        if not documents:
            return success_json({"results": []})

        formatted_results = [
            {
                "title": (metadata.get("title", "Без назви") if metadata else "Без назви"),
                "text": document,
            }
            for document, metadata in zip(documents, metadatas)
        ]

        return success_json({"results": formatted_results})
    except Exception as exc:
        return error_json(f"Не вдалося виконати пошук у базі знань: {exc}")


@mcp.tool()
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Конвертувати суму між валютами за фіксованим довідковим курсом.

    Використовуй цей інструмент, коли користувач хоче перевести бюджет
    подорожі чи вартість готелю з однієї валюти в іншу (EUR, USD, GBP,
    UAH, PLN). Курси — демонстраційні фіксовані значення відносно EUR
    (див. resource travel://currency-rates), а НЕ live-курси біржі.

    Args:
        amount: Сума для конвертації (> 0).
        from_currency: Код вихідної валюти (EUR, USD, GBP, UAH, PLN).
        to_currency: Код цільової валюти (EUR, USD, GBP, UAH, PLN).

    Returns:
        JSON-рядок {"status": "success", "data": {"converted_amount": ...,
        ...}} або {"status": "error", "error": "..."} якщо валюта
        невідома або сума некоректна.
    """

    try:
        if amount <= 0:
            raise ValueError("Сума для конвертації повинна бути більшою за 0.")

        from_currency = from_currency.strip().upper()
        to_currency = to_currency.strip().upper()

        if from_currency not in CURRENCY_RATES_EUR:
            raise ValueError(f"Невідома вихідна валюта: {from_currency}.")
        if to_currency not in CURRENCY_RATES_EUR:
            raise ValueError(f"Невідома цільова валюта: {to_currency}.")

        amount_in_eur = amount / CURRENCY_RATES_EUR[from_currency]
        converted_amount = amount_in_eur * CURRENCY_RATES_EUR[to_currency]

        return success_json(
            {
                "original_amount": amount,
                "from_currency": from_currency,
                "converted_amount": round(converted_amount, 2),
                "to_currency": to_currency,
                "rate_source": "fixed demo rates (travel://currency-rates)",
            }
        )
    except Exception as exc:
        return error_json(f"Не вдалося конвертувати валюту: {exc}")


@mcp.tool()
def book_hotel(hotel_name: str, check_in: str, nights: int, total_cost: float) -> str:
    """Забронювати готель (mock-бронювання).

    РИЗИКОВА ДІЯ: цей tool НЕ можна виконувати без підтвердження людини
    (Human-in-the-Loop). Викликай його лише тоді, коли користувач явно
    просить виконати бронювання, а не просто розрахувати вартість
    (для розрахунку є estimate_hotel_cost). LangGraph-клієнт цього
    MCP-сервера (`mas_langgraph.py`, billing-агент) зупиняє граф через
    `interrupt()` перед фактичним викликом цього tool і продовжує лише
    після explicit approve/reject/edit рішення людини.

    Args:
        hotel_name: Назва готелю (мінімум 2 символи).
        check_in: Дата заїзду у форматі YYYY-MM-DD.
        nights: Кількість ночей (від 1 до 30).
        total_cost: Загальна вартість бронювання у EUR (> 0).

    Returns:
        JSON-рядок {"status": "success", "data": {..., "booking_id": ...}}
        або {"status": "error", "error": "..."} у разі некоректних
        вхідних даних.
    """

    try:
        hotel_name = hotel_name.strip()
        if len(hotel_name) < 2:
            raise ValueError("Назва готелю повинна містити щонайменше 2 символи.")

        check_in = check_in.strip()
        parts = check_in.split("-")
        if len(parts) != 3 or len(parts[0]) != 4 or len(parts[1]) != 2 or len(parts[2]) != 2:
            raise ValueError("Дата повинна бути у форматі YYYY-MM-DD.")

        if not 1 <= nights <= 30:
            raise ValueError("Кількість ночей повинна бути від 1 до 30.")
        if total_cost <= 0:
            raise ValueError("Вартість бронювання повинна бути більшою за 0.")

        return success_json(
            {
                "hotel_name": hotel_name,
                "check_in": check_in,
                "nights": nights,
                "total_cost": total_cost,
                "currency": "EUR",
                "booking_id": "DEMO-BOOKING-001",
            }
        )
    except Exception as exc:
        return error_json(f"Не вдалося забронювати готель: {exc}")


# ================================================================
# RESOURCES (2) — read-only довідники
# ================================================================

@mcp.resource(
    "travel://knowledge-base",
    name="travel_knowledge_index",
    title="Travel knowledge base index",
    description="Список усіх документів внутрішньої бази знань про подорожі (id + title).",
    mime_type="application/json",
)
def travel_knowledge_index() -> str:
    """Індекс документів бази знань (для огляду без виклику search)."""

    import json

    index = [{"id": doc["id"], "title": doc["title"]} for doc in KNOWLEDGE_DOCUMENTS]
    return json.dumps({"documents": index}, ensure_ascii=False, indent=2)


@mcp.resource(
    "travel://currency-rates",
    name="currency_rates",
    title="Currency exchange rates (fixed demo rates)",
    description="Фіксована довідкова таблиця курсів валют відносно EUR, яку використовує convert_currency.",
    mime_type="application/json",
)
def currency_rates() -> str:
    """Таблиця курсів валют, що лежить в основі convert_currency."""

    import json

    return json.dumps(
        {"base_currency": "EUR", "rates": CURRENCY_RATES_EUR}, ensure_ascii=False, indent=2
    )


# ================================================================
# PROMPTS (2) — шаблони, які агент може заповнити
# ================================================================

@mcp.prompt(
    name="trip_planning",
    title="Trip planning prompt",
    description="Шаблон для планування подорожі: агент заповнює параметри і викликає потрібні tools.",
)
def trip_planning_prompt(
    destination: str,
    days: str,
    travelers: str,
    priority: str = "balanced",
) -> str:
    """Формує промпт-шаблон для планування подорожі."""

    return f"""
Сплануй подорож до {destination} на {days} днів для {travelers} осіб.

Пріоритет подорожі: {priority} (cheap / fast / balanced).

Використай доступні MCP tools:
1. recommend_transport — щоб порекомендувати транспорт (потрібна відстань).
2. calculate_trip_budget — щоб розрахувати орієнтовний бюджет.
3. estimate_hotel_cost — щоб оцінити вартість проживання.
4. search_travel_knowledge — щоб перевірити документи, візові вимоги та
   страхування, потрібні для цієї подорожі.

Дай користувачу структуровану відповідь: рекомендований транспорт,
орієнтовний бюджет, вартість готелю та короткий чекліст підготовки.
""".strip()


@mcp.prompt(
    name="budget_summary",
    title="Budget summary prompt",
    description="Шаблон для підсумкового звіту по вже розрахованому бюджету подорожі.",
)
def budget_summary_prompt(trip_budget: str, hotel_cost: str, currency: str = "EUR") -> str:
    """Формує промпт-шаблон для підсумкового звіту по бюджету."""

    return f"""
Ось розраховані витрати на подорож:
- Загальний бюджет подорожі: {trip_budget} {currency}
- Вартість готелю: {hotel_cost} {currency}

Склади короткий підсумковий звіт для користувача: загальна сума витрат,
рекомендований резерв на непередбачені витрати (10-15% від загальної
суми, скористайся search_travel_knowledge для підтвердження цієї
рекомендації), і, якщо користувач попросить іншу валюту — скористайся
tool convert_currency.
""".strip()


# ================================================================
# ENTRYPOINT
# ================================================================

if __name__ == "__main__":
    mcp.run()
