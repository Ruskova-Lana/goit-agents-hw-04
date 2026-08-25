"""ReAct-агент туристичного асистента на LangGraph.

Класичний цикл LLM -> tools -> LLM -> ... -> фінальна відповідь.
На відміну від Plan-and-Execute (plan_execute.py), тут немає окремого
кроку планування: на кожній ітерації LLM сама вирішує, викликати tool
чи одразу відповісти користувачу.

Захисні механізми:
- max_steps       — жорсткий ліміт кількості звернень до LLM;
- timeout         — ліміт часу виконання одного запуску (wall-clock);
- repeat detection — якщо LLM намагається викликати той самий tool
  з тими самими аргументами повторно, виклик блокується і LLM
  отримує підказку не повторюватись; за декілька повторів graph
  примусово завершується;
- JSON-лог траєкторії — кожен крок (виклик LLM, виклик tool,
  спрацювання guardrail) записується у список і зберігається
  у logs/*.json наприкінці запуску.
"""

import asyncio
import json
import operator
import os
import sys
import time

from datetime import datetime, timezone
from typing import Annotated, Literal, TypedDict

# Консоль Windows за замовчуванням використовує cp1252,
# що не підтримує кирилицю у print().
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from tools import calculate_trip_budget, estimate_hotel_cost, recommend_transport
from knowledge import search_knowledge
from tool_utils import error_json, safe_tool_invoke


load_dotenv()


# ================================================================
# Захисні ліміти
# ================================================================

MAX_STEPS = 10
TIMEOUT_SECONDS = 120
MAX_REPEATED_CALLS = 3

LOGS_DIR = "logs"


# ================================================================
# Tools
# ================================================================
# book_hotel (ризиковий tool) навмисно НЕ підключений до ReAct-агента:
# HITL для ризикової дії демонструється в plan_execute.py.
# Тут використовуються лише "безпечні" domain tools + agentic RAG.

REACT_TOOLS = [
    calculate_trip_budget,
    estimate_hotel_cost,
    recommend_transport,
    search_knowledge,
]

TOOLS_BY_NAME = {t.name: t for t in REACT_TOOLS}


SYSTEM_PROMPT = SystemMessage(
    content=(
        "Ти ReAct AI-агент туристичного асистента.\n\n"
        "Працюєш циклічно: думаєш, за потреби викликаєш ОДИН tool, "
        "аналізуєш результат і вирішуєш, чи потрібен ще один tool, "
        "чи можна дати фінальну відповідь користувачу.\n\n"
        "Доступні tools:\n"
        "- calculate_trip_budget — розрахунок загального бюджету подорожі;\n"
        "- estimate_hotel_cost — розрахунок вартості проживання;\n"
        "- recommend_transport — рекомендація транспорту;\n"
        "- search_knowledge — пошук довідкової інформації у внутрішній "
        "ChromaDB knowledge base (страхування, документи, багаж, правила).\n\n"
        "Правила:\n"
        "1. Викликай tool лише тоді, коли він дійсно потрібен для відповіді.\n"
        "2. Ніколи не викликай той самий tool з тими самими аргументами "
        "повторно — якщо результат уже отримано, використай його.\n"
        "3. Коли достатньо інформації — дай коротку фінальну відповідь "
        "користувачу БЕЗ виклику tool.\n"
        "4. Не вигадуй результати tools."
    )
)


# ================================================================
# State
# ================================================================

class ReActState(TypedDict):
    """Стан ReAct агента."""

    # Історія повідомлень (HumanMessage / AIMessage / ToolMessage)
    messages: Annotated[list, operator.add]

    # Кількість звернень до LLM у поточному запуску
    steps: int

    # Час старту запуску (time.monotonic())
    start_time: float

    # JSON-траєкторія виконання
    trajectory: Annotated[list, operator.add]

    # Сигнатури (tool_name, args) вже виконаних викликів — для repeat detection
    call_history: Annotated[list, operator.add]

    # Скільки разів LLM намагалась повторити вже виконаний виклик
    repeat_count: int


# ================================================================
# Допоміжні функції (чисті — легко тестуються без LLM)
# ================================================================

def make_call_signature(name: str, args: dict) -> tuple:
    """Створює хешовану сигнатуру виклику tool для порівняння."""

    return (name, tuple(sorted(args.items())))


def is_repeated_call(call_history: list[tuple], signature: tuple) -> bool:
    """Перевіряє, чи такий виклик tool вже виконувався раніше."""

    return signature in call_history


def elapsed_seconds(start_time: float) -> float:
    """Скільки секунд минуло від старту запуску."""

    return time.monotonic() - start_time


def log_entry(node: str, event: str, **fields) -> dict:
    """Формує один запис JSON-траєкторії."""

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "node": node,
        "event": event,
        **fields,
    }


def save_trajectory(trajectory: list[dict], thread_id: str) -> str:
    """Зберігає JSON-траєкторію виконання у файл logs/*.json."""

    os.makedirs(LOGS_DIR, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(LOGS_DIR, f"react_{thread_id}_{timestamp}.json")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(trajectory, f, ensure_ascii=False, indent=2)

    return path


# ================================================================
# Nodes
# ================================================================

def build_agent_node(llm_with_tools):
    """Створює agent node, прив'язаний до конкретного LLM (для тестів)."""

    def agent_node(state: ReActState) -> dict:
        steps = state.get("steps", 0)
        start_time = state.get("start_time") or time.monotonic()
        trajectory = []

        # --- guardrail: max_steps ---
        if steps >= MAX_STEPS:
            note = AIMessage(
                content=(
                    f"Досягнуто максимальної кількості кроків "
                    f"({MAX_STEPS}). Завершую виконання з наявними "
                    f"результатами."
                )
            )
            trajectory.append(
                log_entry("agent", "max_steps_exceeded", steps=steps)
            )
            return {
                "messages": [note],
                "start_time": start_time,
                "trajectory": trajectory,
            }

        # --- guardrail: timeout ---
        elapsed = elapsed_seconds(start_time)
        if elapsed >= TIMEOUT_SECONDS:
            note = AIMessage(
                content=(
                    f"Перевищено ліміт часу виконання "
                    f"({TIMEOUT_SECONDS}с). Завершую виконання з "
                    f"наявними результатами."
                )
            )
            trajectory.append(
                log_entry("agent", "timeout_exceeded", elapsed=elapsed)
            )
            return {
                "messages": [note],
                "start_time": start_time,
                "trajectory": trajectory,
            }

        messages = [SYSTEM_PROMPT, *state["messages"]]
        response = llm_with_tools.invoke(messages)

        tool_calls = getattr(response, "tool_calls", []) or []

        trajectory.append(
            log_entry(
                "agent",
                "llm_call",
                step=steps + 1,
                content=str(response.content),
                tool_calls=[
                    {"name": c.get("name"), "args": c.get("args")}
                    for c in tool_calls
                ],
            )
        )

        return {
            "messages": [response],
            "steps": steps + 1,
            "start_time": start_time,
            "trajectory": trajectory,
        }

    return agent_node


def tools_node(state: ReActState) -> dict:
    """Виконує усі tool_calls останнього AIMessage з repeat detection."""

    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", []) or []

    call_history = list(state.get("call_history", []))
    repeat_count = state.get("repeat_count", 0)
    trajectory = []
    new_messages = []

    for call in tool_calls:
        name = call.get("name")
        args = call.get("args", {}) or {}
        call_id = call.get("id")
        signature = make_call_signature(name, args)

        if is_repeated_call(call_history, signature):
            repeat_count += 1

            content = (
                "Цей tool вже викликався з такими самими аргументами. "
                "НЕ повторюй виклик — використай попередній результат "
                "і сформуй фінальну відповідь."
            )

            trajectory.append(
                log_entry(
                    "tools",
                    "repeated_call_blocked",
                    tool=name,
                    args=args,
                    repeat_count=repeat_count,
                )
            )

        else:
            tool_function = TOOLS_BY_NAME.get(name)

            if tool_function is None:
                content = error_json(f"Невідомий tool {name}.")
            else:
                # safe_tool_invoke гарантує JSON-контракт навіть при
                # помилці валідації Pydantic args_schema.
                content = safe_tool_invoke(tool_function, args)

            call_history.append(signature)

            trajectory.append(
                log_entry(
                    "tools",
                    "tool_call",
                    tool=name,
                    args=args,
                    result=content,
                )
            )

        new_messages.append(
            ToolMessage(content=content, tool_call_id=call_id, name=name)
        )

    # --- guardrail: забагато повторів підряд -> примусове завершення ---
    if repeat_count >= MAX_REPEATED_CALLS:
        new_messages.append(
            AIMessage(
                content=(
                    "Виявлено повторювані виклики того самого tool. "
                    "Завершую виконання, щоб уникнути нескінченного циклу."
                )
            )
        )
        trajectory.append(
            log_entry("tools", "repeat_limit_exceeded", repeat_count=repeat_count)
        )

    return {
        "messages": new_messages,
        "call_history": call_history,
        "repeat_count": repeat_count,
        "trajectory": trajectory,
    }


def route_after_agent(state: ReActState) -> Literal["tools", "__end__"]:
    """Якщо LLM викликав tool — йдемо в tools, інакше завершуємо."""

    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", []) or []

    if tool_calls:
        return "tools"

    return "__end__"


def route_after_tools(state: ReActState) -> Literal["agent", "__end__"]:
    """Якщо repeat guardrail щойно примусово завершив цикл — END."""

    last_message = state["messages"][-1]

    if isinstance(last_message, AIMessage) and not getattr(
        last_message, "tool_calls", []
    ):
        return "__end__"

    return "agent"


# ================================================================
# Побудова графа (LLM injectable — потрібно для unit-тестів)
# ================================================================

def build_app(chat_model=None):
    """Будує ReAct LangGraph app.

    chat_model — будь-який об'єкт із методами bind_tools()/invoke(),
    сумісний з LangChain chat models. За замовчуванням — Gemini.
    Ін'єкція дозволяє підміняти LLM у тестах фейковою моделлю.
    """

    if chat_model is None:
        from langchain_google_genai import ChatGoogleGenerativeAI

        chat_model = ChatGoogleGenerativeAI(
            model="gemini-3.5-flash-lite",
            temperature=0.1,
        )

    llm_with_tools = chat_model.bind_tools(REACT_TOOLS)

    graph = StateGraph(ReActState)

    graph.add_node("agent", build_agent_node(llm_with_tools))
    graph.add_node("tools", tools_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route_after_agent)
    graph.add_conditional_edges("tools", route_after_tools)

    return graph.compile()


def create_initial_state(query: str) -> ReActState:
    """Початковий стан для нового запуску ReAct-агента."""

    return {
        "messages": [HumanMessage(content=query)],
        "steps": 0,
        "start_time": time.monotonic(),
        "trajectory": [],
        "call_history": [],
        "repeat_count": 0,
    }


# ================================================================
# CLI
# ================================================================

def run_query(query: str, thread_id: str = "react-cli") -> None:
    """Запускає ReAct-агента на одному запиті та зберігає JSON-лог."""

    app = build_app()

    print("\n" + "#" * 80)
    print(f"REACT AGENT — {thread_id}")
    print("#" * 80)
    print(f"USER: {query}")

    result = app.invoke(
        create_initial_state(query),
        config={"recursion_limit": 50},
    )

    print("\n" + "=" * 70)
    print("TRAJECTORY")
    print("=" * 70)

    for entry in result.get("trajectory", []):
        print(entry)

    final_message = result["messages"][-1]

    print("\n" + "=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)
    print(final_message.content)

    log_path = save_trajectory(result.get("trajectory", []), thread_id)
    print(f"\nJSON-лог траєкторії збережено у: {log_path}")


DEMO_QUERIES = [
    (
        "react-demo-budget",
        "Я їду удвох на 5 днів. Щоденний бюджет на одну людину — "
        "80 євро. Порахуй загальний бюджет подорожі.",
    ),
    (
        "react-demo-rag",
        "Що потрібно перевірити перед міжнародною подорожжю? "
        "Скористайся внутрішньою базою знань, якщо потрібно.",
    ),
    (
        "react-demo-multi",
        "Мені потрібно розрахувати вартість готелю на 4 ночі по "
        "100 євро за номер і дізнатись, коли зазвичай відбувається "
        "check-in у готелях.",
    ),
]


def run_demo() -> None:
    """Демонструє ReAct-агента на кількох запитах різної складності."""

    for thread_id, query in DEMO_QUERIES:
        run_query(query, thread_id)


# ================================================================
# ASYNC (додаткова вимога): ainvoke() + паралельне виконання
# ================================================================

async def run_query_async(query: str, thread_id: str = "react-async") -> dict:
    """Асинхронно запускає ReAct-агента через app.ainvoke()."""

    app = build_app()

    return await app.ainvoke(
        create_initial_state(query),
        config={"recursion_limit": 50},
    )


def run_query_async_cli(query: str, thread_id: str = "react-cli-async") -> None:
    """CLI-обгортка над run_query_async з тими самими print/JSON-логами."""

    print("\n" + "#" * 80)
    print(f"REACT AGENT (async) — {thread_id}")
    print("#" * 80)
    print(f"USER: {query}")

    result = asyncio.run(run_query_async(query, thread_id))

    final_message = result["messages"][-1]

    print("\n" + "=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)
    print(final_message.content)

    log_path = save_trajectory(result.get("trajectory", []), thread_id)
    print(f"\nJSON-лог траєкторії збережено у: {log_path}")


async def _run_demo_concurrently() -> float:
    """Запускає всі DEMO_QUERIES ОДНОЧАСНО через asyncio.gather + ainvoke."""

    start = time.monotonic()

    results = await asyncio.gather(
        *[run_query_async(query, thread_id) for thread_id, query in DEMO_QUERIES]
    )

    elapsed = time.monotonic() - start

    for (thread_id, query), result in zip(DEMO_QUERIES, results):
        final_message = result["messages"][-1]
        print(f"\n[{thread_id}] {query}")
        print(f"-> {final_message.content}")

    return elapsed


def run_async_demo() -> None:
    """Порівнює час послідовного (invoke) та паралельного (ainvoke) виконання
    трьох демо-запитів — демонструє практичну вигоду async-виконання."""

    print("\n" + "#" * 80)
    print("ASYNC DEMO — послідовно vs паралельно")
    print("#" * 80)

    print("\n--- Послідовно (app.invoke() у циклі) ---")
    start_sync = time.monotonic()
    for thread_id, query in DEMO_QUERIES:
        app = build_app()
        result = app.invoke(create_initial_state(query), config={"recursion_limit": 50})
        print(f"[{thread_id}] -> {result['messages'][-1].content}")
    elapsed_sync = time.monotonic() - start_sync

    print("\n--- Паралельно (asyncio.gather + app.ainvoke()) ---")
    elapsed_async = asyncio.run(_run_demo_concurrently())

    print("\n" + "=" * 70)
    print("ПОРІВНЯННЯ ЧАСУ ВИКОНАННЯ")
    print("=" * 70)
    print(f"Послідовно (invoke):      {elapsed_sync:.2f} с")
    print(f"Паралельно (ainvoke):     {elapsed_async:.2f} с")

    if elapsed_async > 0:
        print(f"Прискорення:              {elapsed_sync / elapsed_async:.2f}x")


# ================================================================
# ВІЗУАЛІЗАЦІЯ (додаткова вимога): Mermaid-діаграма графа
# ================================================================

def export_graph_diagram() -> str:
    """Зберігає Mermaid-діаграму ReAct-графа у graphs/react_agent.mmd."""

    os.makedirs("graphs", exist_ok=True)

    app = build_app()
    mermaid_text = app.get_graph().draw_mermaid()

    path = os.path.join("graphs", "react_agent.mmd")

    with open(path, "w", encoding="utf-8") as f:
        f.write(mermaid_text)

    print(mermaid_text)
    print(f"\nMermaid-діаграма збережена у: {path}")

    return path


def print_help() -> None:
    print(
        """
============================================================
ReAct Agent — туристичний асистент
============================================================

python react_agent.py run "<запит>"
    Запустити ReAct-агента на довільному запиті (синхронно).

python react_agent.py demo
    Запустити 3 демонстраційні запити.

python react_agent.py arun "<запит>"
    Запустити ReAct-агента асинхронно (app.ainvoke()).

python react_agent.py ademo
    Порівняти час послідовного та паралельного (asyncio.gather)
    виконання 3 демо-запитів.

python react_agent.py graph
    Вивести та зберегти Mermaid-діаграму графа
    (graphs/react_agent.mmd).
============================================================
"""
    )


if __name__ == "__main__":
    command = sys.argv[1].lower() if len(sys.argv) > 1 else "help"

    if command == "run" and len(sys.argv) > 2:
        run_query(sys.argv[2])

    elif command == "demo":
        run_demo()

    elif command == "arun" and len(sys.argv) > 2:
        run_query_async_cli(sys.argv[2])

    elif command == "ademo":
        run_async_demo()

    elif command == "graph":
        export_graph_diagram()

    else:
        print_help()
