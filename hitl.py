"""HITL (Human-in-the-Loop) для ризикових tools — ДЗ4 Завд. 4 (додатково).

Продовження ДЗ2 (`plan_execute.py`, `interrupt_before=["approval"]`) і
Завд. 4 (`guardrails.py`, `tool_guardrail` allowlist per agent): цей модуль
— єдине джерело істини про те, які tools вважаються "ризиковими"
(`RISKY_TOOLS` / `requires_human_approval()`), і надає ГЕНЕРИЧНИЙ вузол
`approval_gate(state, config)`, який можна вставити в БУДЬ-ЯКИЙ LangGraph
tool-calling граф (не лише Plan-and-Execute) — він перехоплює
`tool_calls` останнього `AIMessage`, і для кожного ризикового виклику
зупиняє граф через `interrupt()`, чекаючи рішення людини:

    approve → виконати tool з оригінальними args
    reject  → НЕ виконувати, повернути відмову з причиною
    edit    → виконати tool зі зміненими (merged) args

`approval_gate` підключений до реального MAS-графа (`mas_langgraph.py`,
routing tech/researcher/general-агентів — див. `route_after_agent`
у `build_react_nodes`) — там, де ReAct-агент може запропонувати
ризиковий tool_call (наразі: `book_hotel` у наборі general-агента).
billing-агент (Plan-and-Execute) має власний, окремий HITL-механізм
(`billing_approval_node`, `interrupt_before=["billing_approval"]`) —
обидва підходи демонструють різні, однаково легітимні варіанти HITL у
LangGraph (interrupt() всередині вузла vs. interrupt_before на графі).

Нижче є ДВА демо-графи (agent → approval_gate → END), обидва з 3
сценаріями (approve/reject/edit):

- `python hitl.py demo`     — approval_gate над ЛОКАЛЬНИМ book_hotel
  (LangChain `@tool`, визначений вище в цьому файлі).
- `python hitl.py mcp-demo` — approval_gate над ТИМ САМИМ book_hotel, але
  отриманим через РЕАЛЬНИЙ MCP-протокол (`MultiServerMCPClient` ->
  `mcp_server.py` як stdio subprocess, Завд. 3) — буквальна демонстрація
  "HITL для ризикового MCP-tool", а не просто локальної Python-функції.
"""

import asyncio
import sys
import uuid

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import operator
import sqlite3
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field, field_validator

from guardrails import tool_guardrail
from tool_utils import error_json, safe_tool_invoke, success_json
from tools import calculate_trip_budget


load_dotenv()


# ================================================================
# Pydantic schema для ризикового tool
# ================================================================

class HotelBookingInput(BaseModel):
    """Параметри бронювання готелю."""

    hotel_name: str = Field(description="Назва готелю.")
    check_in: str = Field(description="Дата заїзду у форматі YYYY-MM-DD.")
    nights: int = Field(description="Кількість ночей.")
    total_cost: float = Field(description="Загальна вартість бронювання у EUR.")

    @field_validator("hotel_name")
    @classmethod
    def validate_hotel_name(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 2:
            raise ValueError("Назва готелю повинна містити щонайменше 2 символи.")
        return value

    @field_validator("check_in")
    @classmethod
    def validate_check_in(cls, value: str) -> str:
        value = value.strip()
        parts = value.split("-")
        if len(parts) != 3 or len(parts[0]) != 4 or len(parts[1]) != 2 or len(parts[2]) != 2:
            raise ValueError("Дата повинна бути у форматі YYYY-MM-DD.")
        return value

    @field_validator("nights")
    @classmethod
    def validate_nights(cls, value: int) -> int:
        if not 1 <= value <= 30:
            raise ValueError("Кількість ночей повинна бути від 1 до 30.")
        return value

    @field_validator("total_cost")
    @classmethod
    def validate_total_cost(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("Вартість бронювання повинна бути більшою за 0.")
        return value


# ================================================================
# Ризиковий tool
# ================================================================

@tool(args_schema=HotelBookingInput)
def book_hotel(hotel_name: str, check_in: str, nights: int, total_cost: float) -> str:
    """Забронювати готель.

    РИЗИКОВА ДІЯ: цей tool не можна виконувати без
    підтвердження людини.

    Використовуй цей tool лише тоді, коли користувач
    явно просить виконати бронювання, а не просто
    розрахувати його вартість.

    Args:
        hotel_name: Назва готелю.
        check_in: Дата заїзду у форматі YYYY-MM-DD.
        nights: Кількість ночей.
        total_cost: Загальна вартість у EUR.

    Returns:
        JSON-рядок {"status": "success", "data": {...}} з підтвердженням
        mock-бронювання або {"status": "error", "error": "..."} у разі помилки.
    """

    try:
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
# RISKY TOOLS — єдине джерело істини для всього проєкту
# ================================================================

RISKY_TOOLS: set[str] = {"book_hotel"}
RISKY_TOOLS_BY_NAME = {"book_hotel": book_hotel}


def requires_human_approval(tool_name: str) -> bool:
    """Чи потребує tool явного підтвердження людини (HITL) перед виконанням."""

    return tool_name in RISKY_TOOLS


# ================================================================
# APPROVAL GATE — generic HITL-вузол (interrupt() + Command(resume=...))
# ================================================================

def approval_gate(state: dict, config: RunnableConfig) -> dict:
    """HITL: перехоплює ризикові tool_calls, чекає approve/reject/edit.

    Розрахований на будь-який граф, чий State має принаймні поля
    `messages` (Annotated[list, operator.add]) та (опційно) `current_agent`
    — використовується як для tools_node-заміни у ReAct-агентах
    (`mas_langgraph.py`), так і у самодостатньому демо-графі нижче.

    Для кожного tool_call у останньому AIMessage:
    - НЕ ризиковий  → пропускається без змін (виконає звичайний tools_node
      нижче по графу, якщо він є; у самодостатньому демо тут таких немає).
    - ризиковий, але агенту заборонено ним користуватись (tool_guardrail)
      → одразу відмова, БЕЗ interrupt() (defense-in-depth: людину не варто
      турбувати підтвердженням дії, яку агент і так не має права робити).
    - ризиковий і дозволений → interrupt() зупиняє граф; Command(resume=...)
      з {'action': 'approve'|'reject'|'edit', ...} відновлює виконання.
    """

    last_msg = state["messages"][-1]
    tool_calls = getattr(last_msg, "tool_calls", None) or []
    agent_name = state.get("current_agent", "")

    if not tool_calls:
        return {}

    new_messages: list[ToolMessage] = []

    for tc in tool_calls:
        name = tc["name"]
        args = tc.get("args", {}) or {}
        call_id = tc.get("id")

        if name not in RISKY_TOOLS:
            # Не ризиковий tool_call у тому самому AIMessage — approval_gate
            # його не чіпає; у графах цього проєкту risky/non-risky виклики
            # не змішуються в одному ході, тому пропуск тут безпечний.
            continue

        if not tool_guardrail(agent_name, name):
            content = error_json(
                f"Заборонено guardrail-ом: агенту {agent_name} не дозволено викликати tool {name}."
            )
            new_messages.append(ToolMessage(content=content, tool_call_id=call_id, name=name))
            continue

        decision = interrupt(
            {
                "message": "Підтвердити ризикову дію",
                "tool": name,
                "args": args,
                "agent_name": agent_name,
            }
        )

        action = str((decision or {}).get("action", "reject")).lower()
        tool_function = RISKY_TOOLS_BY_NAME[name]

        if action == "approve":
            result = safe_tool_invoke(tool_function, args)
            content = f"{name}: {result}"
        elif action == "edit":
            edited_args = {**args, **(decision.get("args") or {})}
            result = safe_tool_invoke(tool_function, edited_args)
            content = f"{name} (параметри змінено): {result}"
        else:
            reason = str((decision or {}).get("reason", ""))
            content = f"Дія {name} відхилена."
            if reason:
                content += f" Причина: {reason}"

        new_messages.append(ToolMessage(content=content, tool_call_id=call_id, name=name))

    return {"messages": new_messages}


# ================================================================
# Самодостатній демо-граф: agent → approval_gate → END
# ================================================================

DEMO_TOOLS = [book_hotel, calculate_trip_budget]
DEMO_TOOLS_BY_NAME = {t.name: t for t in DEMO_TOOLS}

DEMO_SYSTEM_PROMPT = (
    "Ти демо-агент бронювання готелів. Якщо користувач просить виконати "
    "бронювання — викликай book_hotel. Якщо просить лише розрахувати "
    "вартість — викликай calculate_trip_budget. Один tool за раз."
)


class HitlDemoState(TypedDict):
    messages: Annotated[list, operator.add]
    current_agent: str


def _build_demo_llm():
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0.1)
    return llm.bind_tools(DEMO_TOOLS)


def demo_agent_node(state: HitlDemoState) -> dict:
    llm_with_tools = _build_demo_llm()
    response = llm_with_tools.invoke([SystemMessage(content=DEMO_SYSTEM_PROMPT), *state["messages"]])
    return {"messages": [response]}


def demo_tools_node(state: HitlDemoState) -> dict:
    """Виконує НЕ ризикові tool_calls (approval_gate уже обробив ризикові)."""

    last_msg = state["messages"][-1]
    tool_calls = getattr(last_msg, "tool_calls", None) or []
    new_messages = []

    for tc in tool_calls:
        name = tc["name"]
        if name in RISKY_TOOLS:
            continue
        tool_function = DEMO_TOOLS_BY_NAME.get(name)
        content = (
            safe_tool_invoke(tool_function, tc.get("args", {}) or {})
            if tool_function is not None
            else error_json(f"Невідомий tool {name}.")
        )
        new_messages.append(ToolMessage(content=content, tool_call_id=tc.get("id"), name=name))

    return {"messages": new_messages} if new_messages else {}


def route_after_demo_agent(state: HitlDemoState) -> Literal["approval_gate", "__end__"]:
    last_msg = state["messages"][-1]
    tool_calls = getattr(last_msg, "tool_calls", None) or []
    return "approval_gate" if tool_calls else "__end__"


def route_after_approval_gate(state: HitlDemoState) -> Literal["demo_tools", "__end__"]:
    # Знаходимо останній AIMessage з tool_calls (а не фіксований offset —
    # approval_gate міг додати 0..N ToolMessage залежно від кількості
    # ризикових викликів у цьому ході).
    last_ai_message = next(
        (m for m in reversed(state["messages"]) if isinstance(m, AIMessage)), None
    )
    original_tool_calls = getattr(last_ai_message, "tool_calls", None) or []
    unhandled_non_risky = [tc for tc in original_tool_calls if tc["name"] not in RISKY_TOOLS]
    return "demo_tools" if unhandled_non_risky else "__end__"


demo_graph = StateGraph(HitlDemoState)
demo_graph.add_node("agent", demo_agent_node)
demo_graph.add_node("approval_gate", approval_gate)
demo_graph.add_node("demo_tools", demo_tools_node)

demo_graph.add_edge(START, "agent")
demo_graph.add_conditional_edges("agent", route_after_demo_agent)
demo_graph.add_conditional_edges("approval_gate", route_after_approval_gate)
demo_graph.add_edge("demo_tools", END)

demo_connection = sqlite3.connect("hitl_demo_state.db", check_same_thread=False)
demo_saver = SqliteSaver(demo_connection)
demo_app = demo_graph.compile(checkpointer=demo_saver)


def _demo_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _initial_demo_state(query: str) -> HitlDemoState:
    return {"messages": [HumanMessage(content=query)], "current_agent": "billing"}


def _print_interrupt(result: dict) -> None:
    interrupts = result.get("__interrupt__", [])
    for item in interrupts:
        print(f"INTERRUPT PAYLOAD: {getattr(item, 'value', item)}")


DEMO_QUERY = (
    "Я хочу забронювати Demo Travel Hotel з 2026-09-15 на 4 ночі. "
    "Загальна вартість — 400 євро. Виконай бронювання."
)


def demo_approve() -> None:
    """Сценарій 1: interrupt() → Command(resume={'action': 'approve'})."""

    print("\n" + "#" * 80)
    print("HITL DEMO — SCENARIO 1: APPROVE")
    print("#" * 80)

    thread_id = f"hitl-demo-approve-{uuid.uuid4().hex[:8]}"
    config = _demo_config(thread_id)

    result = demo_app.invoke(_initial_demo_state(DEMO_QUERY), config=config)
    _print_interrupt(result)

    result = demo_app.invoke(Command(resume={"action": "approve"}), config=config)
    print(f"\nFINAL MESSAGES:")
    for m in result["messages"]:
        if isinstance(m, ToolMessage):
            print(f"  [tool:{m.name}] {m.content}")


def demo_reject() -> None:
    """Сценарій 2: interrupt() → Command(resume={'action': 'reject', 'reason': ...})."""

    print("\n" + "#" * 80)
    print("HITL DEMO — SCENARIO 2: REJECT")
    print("#" * 80)

    thread_id = f"hitl-demo-reject-{uuid.uuid4().hex[:8]}"
    config = _demo_config(thread_id)

    result = demo_app.invoke(_initial_demo_state(DEMO_QUERY), config=config)
    _print_interrupt(result)

    result = demo_app.invoke(
        Command(resume={"action": "reject", "reason": "Клієнт скасував бронювання."}),
        config=config,
    )
    print(f"\nFINAL MESSAGES:")
    for m in result["messages"]:
        if isinstance(m, ToolMessage):
            print(f"  [tool:{m.name}] {m.content}")


def demo_edit() -> None:
    """Сценарій 3: interrupt() → Command(resume={'action': 'edit', 'args': {...}})."""

    print("\n" + "#" * 80)
    print("HITL DEMO — SCENARIO 3: EDIT")
    print("#" * 80)

    thread_id = f"hitl-demo-edit-{uuid.uuid4().hex[:8]}"
    config = _demo_config(thread_id)

    result = demo_app.invoke(_initial_demo_state(DEMO_QUERY), config=config)
    _print_interrupt(result)

    result = demo_app.invoke(
        Command(resume={"action": "edit", "args": {"nights": 3, "total_cost": 300}}),
        config=config,
    )
    print(f"\nFINAL MESSAGES:")
    for m in result["messages"]:
        if isinstance(m, ToolMessage):
            print(f"  [tool:{m.name}] {m.content}")


def run_demo() -> None:
    demo_approve()
    demo_reject()
    demo_edit()


# ================================================================
# HITL на РЕАЛЬНОМУ MCP-tool (book_hotel через MultiServerMCPClient)
# ================================================================
#
# approval_gate() вище виконує ризикові tools через safe_tool_invoke()
# (синхронний виклик локальної LangChain @tool-функції). Tools, отримані
# від MCP-сервера через langchain-mcp-adapters, — це async-only
# StructuredTool (їхній .invoke() навмисно кидає NotImplementedError:
# "StructuredTool does not support sync invocation") — тому для
# демонстрації HITL САМЕ на MCP-tool потрібен окремий, async-варіант
# approval_gate, що викликає tool_function.ainvoke(args) замість
# safe_tool_invoke(). Механізм HITL (interrupt()/Command(resume=...),
# approve/reject/edit) — той самий; відрізняється лише спосіб виконання
# самого tool.

MCP_DEMO_DB_PATH = "hitl_mcp_demo_state.db"


def _extract_mcp_result_text(result) -> str:
    """MCP tool.ainvoke() повертає список content-блоків
    [{'type': 'text', 'text': '...json...', 'id': ...}], а не рядок —
    дістаємо текст для читабельного виводу/ToolMessage.content."""

    if isinstance(result, list):
        parts = [
            block.get("text", "")
            for block in result
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "\n".join(parts) if parts else str(result)
    return str(result)


def build_mcp_approval_gate(risky_tools_by_name: dict):
    """Factory: та сама HITL-логіка, що й approval_gate(), але виконує
    ризиковий tool через РЕАЛЬНИЙ MCP-виклик (tool.ainvoke()) — тому
    сам approval_gate тут async-вузол (LangGraph підтримує це нативно
    через app.ainvoke())."""

    async def approval_gate_mcp(state: dict, config: RunnableConfig) -> dict:
        last_msg = state["messages"][-1]
        tool_calls = getattr(last_msg, "tool_calls", None) or []
        agent_name = state.get("current_agent", "")

        if not tool_calls:
            return {}

        new_messages: list[ToolMessage] = []

        for tc in tool_calls:
            name = tc["name"]
            args = tc.get("args", {}) or {}
            call_id = tc.get("id")

            if name not in RISKY_TOOLS:
                continue

            if not tool_guardrail(agent_name, name):
                content = error_json(
                    f"Заборонено guardrail-ом: агенту {agent_name} не дозволено викликати tool {name}."
                )
                new_messages.append(ToolMessage(content=content, tool_call_id=call_id, name=name))
                continue

            decision = interrupt(
                {
                    "message": "Підтвердити ризикову дію (реальний MCP tool)",
                    "tool": name,
                    "args": args,
                    "agent_name": agent_name,
                }
            )

            action = str((decision or {}).get("action", "reject")).lower()
            mcp_tool = risky_tools_by_name[name]

            if action == "approve":
                raw_result = await mcp_tool.ainvoke(args)
                content = f"{name}: {_extract_mcp_result_text(raw_result)}"
            elif action == "edit":
                edited_args = {**args, **(decision.get("args") or {})}
                raw_result = await mcp_tool.ainvoke(edited_args)
                content = f"{name} (параметри змінено): {_extract_mcp_result_text(raw_result)}"
            else:
                reason = str((decision or {}).get("reason", ""))
                content = f"Дія {name} відхилена."
                if reason:
                    content += f" Причина: {reason}"

            new_messages.append(ToolMessage(content=content, tool_call_id=call_id, name=name))

        return {"messages": new_messages}

    return approval_gate_mcp


def route_after_mcp_approval_gate(state: HitlDemoState) -> Literal["mcp_tools", "__end__"]:
    last_ai_message = next(
        (m for m in reversed(state["messages"]) if isinstance(m, AIMessage)), None
    )
    original_tool_calls = getattr(last_ai_message, "tool_calls", None) or []
    unhandled_non_risky = [tc for tc in original_tool_calls if tc["name"] not in RISKY_TOOLS]
    return "mcp_tools" if unhandled_non_risky else "__end__"


async def _build_mcp_demo_app(saver: AsyncSqliteSaver):
    """Піднімає mcp_server.py як stdio subprocess (MultiServerMCPClient,
    Завд. 3), будує agent → approval_gate_mcp → mcp_tools граф над
    РЕАЛЬНИМИ MCP tools.

    Async-граф (mcp_agent_node/approval_gate_mcp — async def) потребує
    AsyncSqliteSaver замість звичайного SqliteSaver (той підтримує лише
    синхронний API — app.invoke(), не app.ainvoke())."""

    client = MultiServerMCPClient(
        {"travel": {"transport": "stdio", "command": sys.executable, "args": ["mcp_server.py"]}}
    )
    mcp_tools = await client.get_tools()
    mcp_tools_by_name = {t.name: t for t in mcp_tools}
    risky_mcp_tools_by_name = {
        name: mcp_tools_by_name[name] for name in RISKY_TOOLS if name in mcp_tools_by_name
    }

    llm_with_mcp_tools = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite", temperature=0.1
    ).bind_tools(mcp_tools)

    async def mcp_agent_node(state: HitlDemoState) -> dict:
        response = await llm_with_mcp_tools.ainvoke(
            [SystemMessage(content=DEMO_SYSTEM_PROMPT), *state["messages"]]
        )
        return {"messages": [response]}

    async def mcp_tools_node(state: HitlDemoState) -> dict:
        """Виконує НЕ ризикові MCP tool_calls (approval_gate_mcp обробив ризикові)."""

        last_msg = state["messages"][-1]
        tool_calls = getattr(last_msg, "tool_calls", None) or []
        new_messages = []

        for tc in tool_calls:
            name = tc["name"]
            if name in RISKY_TOOLS:
                continue
            tool_fn = mcp_tools_by_name.get(name)
            if tool_fn is None:
                content = error_json(f"Невідомий tool {name}.")
            else:
                raw_result = await tool_fn.ainvoke(tc.get("args", {}) or {})
                content = _extract_mcp_result_text(raw_result)
            new_messages.append(ToolMessage(content=content, tool_call_id=tc.get("id"), name=name))

        return {"messages": new_messages} if new_messages else {}

    mcp_graph = StateGraph(HitlDemoState)
    mcp_graph.add_node("agent", mcp_agent_node)
    mcp_graph.add_node("approval_gate", build_mcp_approval_gate(risky_mcp_tools_by_name))
    mcp_graph.add_node("mcp_tools", mcp_tools_node)

    mcp_graph.add_edge(START, "agent")
    mcp_graph.add_conditional_edges("agent", route_after_demo_agent)
    mcp_graph.add_conditional_edges("approval_gate", route_after_mcp_approval_gate)
    mcp_graph.add_edge("mcp_tools", END)

    return mcp_graph.compile(checkpointer=saver)


async def _run_mcp_scenario(app, thread_id: str, resume_value: dict, label: str) -> None:
    config = _demo_config(thread_id)

    result = await app.ainvoke(_initial_demo_state(DEMO_QUERY), config=config)
    _print_interrupt(result)

    result = await app.ainvoke(Command(resume=resume_value), config=config)

    print(f"\nFINAL MESSAGES ({label}, через реальний MCP-виклик):")
    for m in result["messages"]:
        if isinstance(m, ToolMessage):
            print(f"  [tool:{m.name}] {m.content}")


async def run_mcp_demo() -> None:
    """3 сценарії (approve/reject/edit) на РИЗИКОВОМУ MCP-tool book_hotel,
    отриманому через MultiServerMCPClient з mcp_server.py (Завд. 3)."""

    print("\n" + "#" * 80)
    print("HITL DEMO — RISKY MCP-TOOL (book_hotel через MultiServerMCPClient)")
    print("#" * 80)

    run_id = uuid.uuid4().hex[:8]

    async with AsyncSqliteSaver.from_conn_string(MCP_DEMO_DB_PATH) as saver:
        app = await _build_mcp_demo_app(saver)

        print("\n" + "#" * 80)
        print("SCENARIO 1: APPROVE (MCP)")
        print("#" * 80)
        await _run_mcp_scenario(app, f"hitl-mcp-demo-approve-{run_id}", {"action": "approve"}, "APPROVE")

        print("\n" + "#" * 80)
        print("SCENARIO 2: REJECT (MCP)")
        print("#" * 80)
        await _run_mcp_scenario(
            app,
            f"hitl-mcp-demo-reject-{run_id}",
            {"action": "reject", "reason": "Клієнт скасував бронювання."},
            "REJECT",
        )

        print("\n" + "#" * 80)
        print("SCENARIO 3: EDIT (MCP)")
        print("#" * 80)
        await _run_mcp_scenario(
            app,
            f"hitl-mcp-demo-edit-{run_id}",
            {"action": "edit", "args": {"nights": 3, "total_cost": 300}},
            "EDIT",
        )


def print_help() -> None:
    print(
        """
============================================================
hitl.py — approval_gate demo (ДЗ4 Завд. 4, додатково)
============================================================

python hitl.py demo        Усі 3 сценарії на ЛОКАЛЬНОМУ book_hotel.
python hitl.py approve     Лише сценарій approve (локальний).
python hitl.py reject      Лише сценарій reject (локальний).
python hitl.py edit        Лише сценарій edit (локальний).

python hitl.py mcp-demo    Усі 3 сценарії (approve/reject/edit) на
                            РИЗИКОВОМУ MCP-tool book_hotel — піднімає
                            mcp_server.py як stdio subprocess через
                            MultiServerMCPClient (Завд. 3) і виконує
                            approve/reject/edit через РЕАЛЬНИЙ
                            tool.ainvoke() MCP-виклик.
============================================================
"""
    )


if __name__ == "__main__":
    command = sys.argv[1].lower() if len(sys.argv) > 1 else "help"

    if command == "demo":
        run_demo()
    elif command == "approve":
        demo_approve()
    elif command == "reject":
        demo_reject()
    elif command == "edit":
        demo_edit()
    elif command == "mcp-demo":
        asyncio.run(run_mcp_demo())
    else:
        print_help()
