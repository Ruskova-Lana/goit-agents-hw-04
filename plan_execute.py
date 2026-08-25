import operator
import os
import sqlite3
import sys

from typing import Annotated, Literal, TypedDict

# Консоль Windows за замовчуванням використовує cp1252,
# що не підтримує кирилицю у print().
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

from tools import (
    calculate_trip_budget,
    estimate_hotel_cost,
    recommend_transport,
)

from knowledge import search_knowledge
from hitl import book_hotel
from tool_utils import safe_tool_invoke


# ================================================================
# Завантаження змінних середовища
# ================================================================

load_dotenv()


# ================================================================
# Structured Output Models
# ================================================================

class Plan(BaseModel):
    """Структурований план виконання задачі."""

    goal: str = Field(
        description="Головна ціль користувацького запиту."
    )

    steps: list[str] = Field(
        description=(
            "Послідовний список конкретних кроків "
            "для досягнення цілі."
        )
    )


class ReplanDecision(BaseModel):
    """Рішення replanner після виконання чергового кроку."""

    action: Literal[
        "continue",
        "replan",
        "finish",
    ] = Field(
        description=(
            "continue — виконати наступний крок; "
            "replan — змінити залишковий план; "
            "finish — завершити виконання."
        )
    )

    updated_steps: list[str] | None = Field(
        default=None,
        description=(
            "Новий список невиконаних кроків, "
            "якщо action = replan."
        ),
    )

    reasoning: str = Field(
        description="Коротке пояснення рішення replanner."
    )


# ================================================================
# State
# ================================================================

class PlanExecuteState(TypedDict):
    """Стан Plan-and-Execute агента."""

    # Історія повідомлень
    messages: Annotated[list, operator.add]

    # Поточний план
    plan: list[str]

    # Індекс наступного кроку
    current_step: int

    # Результати виконаних кроків
    results: list[str]

    # Чи завершено задачу
    completed: bool

    # Головна ціль, сформована Planner
    goal: str

    # Tool call, який очікує HITL approval
    pending_tool_call: dict | None

    # Рішення людини (approve/reject/edit), записане через
    # app.update_state() перед відновленням після interrupt_before
    human_decision: dict | None

    # Demo-пауза для перевірки persistence
    pause_after_first_step: bool

    # Чи demo-пауза вже була виконана
    pause_done: bool


# ================================================================
# Google Gemini
# ================================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0.1,
)


# ================================================================
# Structured LLMs
# ================================================================

planner_llm = llm.with_structured_output(
    Plan
)

replanner_llm = llm.with_structured_output(
    ReplanDecision
)


# ================================================================
# Tools
# ================================================================

# Звичайні tools + Agentic RAG + ризиковий HITL tool
tools = [
    calculate_trip_budget,
    estimate_hotel_cost,
    recommend_transport,
    search_knowledge,
    book_hotel,
]


tools_by_name = {
    current_tool.name: current_tool
    for current_tool in tools
}


# Tools, які не можна виконувати без підтвердження людини
RISKY_TOOLS = {
    "book_hotel"
}


# LLM самостійно вирішує, який tool потрібний
executor_llm = llm.bind_tools(
    tools
)


# ================================================================
# Допоміжні функції
# ================================================================

def get_user_query(
    state: PlanExecuteState,
) -> str:
    """Повертає початковий запит користувача."""

    for message in state["messages"]:
        if isinstance(message, HumanMessage):
            return str(message.content)

    return ""


def format_results(
    results: list[str],
) -> str:
    """Перетворює попередні результати у текст для prompt."""

    if not results:
        return "Попередніх результатів немає."

    return "\n".join(
        f"- {result}"
        for result in results
    )


# ================================================================
# PLANNER
# ================================================================

def planner_node(
    state: PlanExecuteState,
) -> dict:
    """Генерує структурований план виконання задачі."""

    user_query = get_user_query(
        state
    )


    prompt = f"""
Ти Planner туристичного AI-агента.

Запит користувача:
{user_query}

Створи структурований план виконання задачі.

Доступні інструменти:

1. calculate_trip_budget
   Розрахунок загального бюджету подорожі.

2. estimate_hotel_cost
   Розрахунок вартості проживання в готелі.

3. recommend_transport
   Рекомендація транспорту залежно від відстані
   та пріоритету користувача.

4. search_knowledge
   Пошук у внутрішній ChromaDB knowledge base.
   Використовуй для довідкової інформації:
   документи, страхування, багаж, правила подорожей,
   hotel policies та travel recommendations.

5. book_hotel
   Фактичне бронювання готелю.
   Це РИЗИКОВА дія, яка потребує Human-in-the-Loop approval.

Правила планування:

- сформуй від 1 до 5 конкретних кроків;
- один крок = одна логічна дія;
- у кожному кроці явно вкажи очікуваний tool;
- не виконуй розрахунки самостійно;
- не додавай search_knowledge для простих розрахунків;
- додавай book_hotel ТІЛЬКИ якщо користувач явно просить
  виконати бронювання;
- не додавай зайвих кроків.
"""


    plan_result = planner_llm.invoke(
        prompt
    )


    print("\n" + "=" * 70)
    print("PLANNER")
    print("=" * 70)

    print(
        f"Goal: {plan_result.goal}"
    )

    print("\nPlan:")

    for index, step in enumerate(
        plan_result.steps,
        start=1,
    ):
        print(
            f"{index}. {step}"
        )


    return {
        "goal": plan_result.goal,
        "plan": plan_result.steps,
        "current_step": 0,
        "results": [],
        "completed": False,
        "pending_tool_call": None,

        "messages": [
            AIMessage(
                content=(
                    f"Planner створив план: "
                    f"{plan_result.steps}"
                )
            )
        ],
    }


# ================================================================
# EXECUTOR
# ================================================================

def executor_node(
    state: PlanExecuteState,
) -> dict:
    """Виконує рівно один крок поточного плану."""

    step_index = state[
        "current_step"
    ]

    plan = state[
        "plan"
    ]


    if step_index >= len(plan):
        return {
            "completed": True
        }


    current_step_text = plan[
        step_index
    ]

    user_query = get_user_query(
        state
    )


    print("\n" + "-" * 70)

    print(
        f"EXECUTOR — STEP {step_index + 1}"
    )

    print("-" * 70)

    print(
        f"Current step: {current_step_text}"
    )


    prompt = f"""
Ти Executor туристичного Plan-and-Execute агента.

Оригінальний запит користувача:
{user_query}

Поточний крок плану:
{current_step_text}

Попередні результати:
{format_results(state["results"])}

Виконай ТІЛЬКИ цей один крок.

Доступні tools:

- calculate_trip_budget
- estimate_hotel_cost
- recommend_transport
- search_knowledge
- book_hotel

Правила:

1. Самостійно вибери tool, який відповідає поточному кроку.

2. Використовуй calculate_trip_budget для розрахунку
   бюджету подорожі.

3. Використовуй estimate_hotel_cost для розрахунку
   вартості проживання.

4. Використовуй recommend_transport для рекомендації
   транспорту.

5. Використовуй search_knowledge ТІЛЬКИ коли потрібні
   довідкові знання з внутрішньої бази ChromaDB:
   документи, страхування, багаж, правила,
   рекомендації та hotel policies.

6. НЕ використовуй search_knowledge для арифметичних
   розрахунків.

7. Використовуй book_hotel ТІЛЬКИ якщо поточний крок
   явно вимагає фактичного бронювання.

8. Використовуй максимум один tool для цього кроку.

9. Не переходь до наступного кроку.

10. Не вигадуй результат виконання tool.
"""


    response = executor_llm.invoke(
        prompt
    )


    tool_calls = getattr(
        response,
        "tool_calls",
        [],
    )


    # ============================================================
    # Якщо LLM не викликав tool
    # ============================================================

    if not tool_calls:

        result_text = str(
            response.content
        )


        print(
            f"Result without tool: {result_text}"
        )


        return {
            "current_step": step_index + 1,

            "results": [
                *state["results"],
                (
                    f"Крок {step_index + 1}: "
                    f"{result_text}"
                ),
            ],

            "messages": [
                AIMessage(
                    content=(
                        f"Виконано крок "
                        f"{step_index + 1}: "
                        f"{result_text}"
                    )
                )
            ],
        }


    # ============================================================
    # Беремо перший tool call
    # Один executor iteration = один logical step
    # ============================================================

    tool_call = tool_calls[0]

    tool_name = tool_call.get(
        "name"
    )

    tool_args = tool_call.get(
        "args",
        {},
    )


    print(
        f"Selected tool: {tool_name}"
    )

    print(
        f"Arguments: {tool_args}"
    )


    tool_function = tools_by_name.get(
        tool_name
    )


    if tool_function is None:

        result_text = (
            f"Помилка: невідомий tool {tool_name}."
        )


        return {
            "current_step": step_index + 1,

            "results": [
                *state["results"],
                (
                    f"Крок {step_index + 1}: "
                    f"{result_text}"
                ),
            ],

            "messages": [
                AIMessage(
                    content=result_text
                )
            ],
        }


    # ============================================================
    # HITL: ризиковий tool НЕ виконуємо одразу
    # ============================================================

    if tool_name in RISKY_TOOLS:

        print(
            "\nRisky tool detected. "
            "Human approval is required."
        )


        # Зберігаємо tool call у State.
        # Наступним node буде approval_node.
        return {
            "pending_tool_call": {
                "name": tool_name,
                "args": tool_args,
                "step_index": step_index,
                "step_text": current_step_text,
            },

            "messages": [
                AIMessage(
                    content=(
                        f"Ризиковий tool {tool_name} "
                        f"очікує підтвердження користувача."
                    )
                )
            ],
        }


    # ============================================================
    # Звичайний tool виконуємо одразу
    # safe_tool_invoke гарантує JSON-контракт навіть при помилці
    # ============================================================

    tool_result = safe_tool_invoke(
        tool_function,
        tool_args,
    )


    result_text = (
        f"{tool_name}: {tool_result}"
    )


    print(
        f"Result: {result_text}"
    )


    return {
        "current_step": step_index + 1,

        "results": [
            *state["results"],
            (
                f"Крок {step_index + 1}: "
                f"{result_text}"
            ),
        ],

        "messages": [
            AIMessage(
                content=(
                    f"Виконано крок "
                    f"{step_index + 1}: "
                    f"{result_text}"
                )
            )
        ],
    }


# ================================================================
# ROUTER ПІСЛЯ EXECUTOR
# ================================================================

def route_after_executor(
    state: PlanExecuteState,
) -> Literal[
    "approval",
    "checkpoint_pause",
    "replanner",
]:
    """Визначає наступний node після Executor."""


    # Якщо Executor знайшов risky tool
    if state.get(
        "pending_tool_call"
    ):
        return "approval"


    # Для демонстрації persistence:
    # після першого виконаного кроку
    # навмисно зупиняємо workflow
    if (
        state.get(
            "pause_after_first_step",
            False,
        )
        and not state.get(
            "pause_done",
            False,
        )
        and state.get(
            "current_step",
            0,
        ) >= 1
    ):
        return "checkpoint_pause"


    return "replanner"


# ================================================================
# CHECKPOINT PAUSE
# ================================================================

def checkpoint_pause_node(
    state: PlanExecuteState,
) -> dict:
    """Навмисно зупиняє graph для демонстрації persistence.

    State вже збережений SqliteSaver.
    Після нового запуску процесу workflow продовжується
    через Command(resume=...) з тим самим thread_id.
    """


    resume_value = interrupt(
        {
            "type": "checkpoint_demo",

            "message": (
                "Workflow навмисно призупинено "
                "після першого виконаного кроку."
            ),

            "current_step": state[
                "current_step"
            ],

            "plan": state[
                "plan"
            ],

            "results": state[
                "results"
            ],

            "instruction": (
                "Перезапустіть процес і використайте "
                "той самий thread_id для resume."
            ),
        }
    )


    print(
        f"\nCheckpoint resumed: {resume_value}"
    )


    return {
        "pause_done": True,
        "pause_after_first_step": False,
    }


# ================================================================
# HUMAN-IN-THE-LOOP APPROVAL
# ================================================================
#
# Граф компілюється з interrupt_before=["approval"] (див. секцію
# SQLITE CHECKPOINTER нижче), тому виконання ЗУПИНЯЄТЬСЯ ще ДО того,
# як цей node почне виконуватись. Людина бачить деталі ризикової дії
# через app.get_state(config).values["pending_tool_call"], записує своє
# рішення через app.update_state(config, {"human_decision": {...}}),
# і лише тоді app.invoke(None, config) відновлює граф — approval_node
# запускається з уже заповненим state["human_decision"].

def approval_node(
    state: PlanExecuteState,
) -> dict:
    """Виконує (або відхиляє) ризиковий tool за рішенням людини.

    Рішення (approve / reject / edit) вже мало бути записане у
    state["human_decision"] ЗОВНІ, до відновлення graph після
    interrupt_before. Підтримує:
    - approve
    - reject
    - edit
    """

    pending = state.get(
        "pending_tool_call"
    )


    if not pending:

        return {}


    tool_name = pending[
        "name"
    ]

    original_args = pending[
        "args"
    ]

    step_index = pending[
        "step_index"
    ]


    # ============================================================
    # Рішення людини (записане через app.update_state() ДО resume)
    # ============================================================

    decision = state.get(
        "human_decision"
    ) or {
        "action": "reject",
        "reason": "Рішення людини не було надано.",
    }

    action = str(
        decision.get(
            "action",
            "reject",
        )
    ).lower()


    tool_function = tools_by_name[
        tool_name
    ]


    # ============================================================
    # APPROVE
    # ============================================================

    if action == "approve":

        tool_result = safe_tool_invoke(
            tool_function,
            original_args,
        )


        result_text = (
            f"{tool_name}: {tool_result}"
        )


    # ============================================================
    # EDIT
    # ============================================================

    elif action == "edit":

        edited_args = decision.get(
            "args"
        )


        if not edited_args:

            result_text = (
                "Edit відхилено: "
                "поле args відсутнє."
            )

        else:

            # safe_tool_invoke гарантує JSON-контракт навіть
            # при помилці Pydantic-валідації edited_args.
            tool_result = safe_tool_invoke(
                tool_function,
                edited_args,
            )


            result_text = (
                f"{tool_name} "
                f"(параметри змінено): "
                f"{tool_result}"
            )


    # ============================================================
    # REJECT
    # ============================================================

    else:

        reason = str(
            decision.get(
                "reason",
                "",
            )
        )


        result_text = (
            "Ризикову дію відхилено користувачем."
        )


        if reason:

            result_text += (
                f" Причина: {reason}"
            )


    print("\n" + "=" * 70)
    print("HITL DECISION")
    print("=" * 70)

    print(
        f"Action: {action}"
    )

    print(
        f"Result: {result_text}"
    )


    return {
        "current_step": step_index + 1,

        "pending_tool_call": None,

        "human_decision": None,

        "results": [
            *state["results"],
            (
                f"Крок {step_index + 1}: "
                f"{result_text}"
            ),
        ],

        "messages": [
            AIMessage(
                content=(
                    f"HITL result: "
                    f"{result_text}"
                )
            )
        ],
    }


# ================================================================
# REPLANNER
# ================================================================

def replanner_node(
    state: PlanExecuteState,
) -> dict:
    """Вирішує: continue, replan або finish."""

    plan = state[
        "plan"
    ]

    current_step = state[
        "current_step"
    ]

    results = state[
        "results"
    ]

    remaining_steps = plan[
        current_step:
    ]


    print("\n" + "-" * 70)
    print("REPLANNER")
    print("-" * 70)


    prompt = f"""
Ти Replanner туристичного Plan-and-Execute агента.

Головна ціль:
{state.get("goal", "")}

Початковий запит:
{get_user_query(state)}

Поточний план:
{plan}

Виконано кроків:
{current_step} із {len(plan)}

Результати:
{format_results(results)}

Залишкові кроки:
{remaining_steps}

Прийми одне рішення:

continue
- якщо план залишається правильним
  і потрібно виконати наступний крок.

replan
- якщо після отриманого результату
  залишковий план необхідно змінити;
- updated_steps повинен містити ТІЛЬКИ
  нові невиконані кроки.

finish
- якщо ціль користувача вже досягнута;
- якщо всі кроки виконані, вибери finish.

Не повторюй уже виконані кроки.
"""


    decision = replanner_llm.invoke(
        prompt
    )


    print(
        f"Decision: {decision.action}"
    )

    print(
        f"Reasoning: {decision.reasoning}"
    )


    # ============================================================
    # FINISH
    # ============================================================

    if decision.action == "finish":

        return {
            "completed": True,

            "messages": [
                AIMessage(
                    content=(
                        f"Задачу завершено. "
                        f"{decision.reasoning}"
                    )
                )
            ],
        }


    # ============================================================
    # REPLAN
    # ============================================================

    if (
        decision.action == "replan"
        and decision.updated_steps
    ):

        print("\nUpdated remaining plan:")

        for index, step in enumerate(
            decision.updated_steps,
            start=1,
        ):

            print(
                f"{index}. {step}"
            )


        # updated_steps містить тільки
        # невиконану частину плану.
        # Results попередніх кроків зберігаються.
        return {
            "plan": decision.updated_steps,
            "current_step": 0,

            "messages": [
                AIMessage(
                    content=(
                        f"Replanner змінив залишковий план: "
                        f"{decision.updated_steps}"
                    )
                )
            ],
        }


    # ============================================================
    # Додатковий safeguard:
    # кроків більше немає
    # ============================================================

    if current_step >= len(plan):

        return {
            "completed": True,

            "messages": [
                AIMessage(
                    content=(
                        "Усі кроки плану виконані."
                    )
                )
            ],
        }


    # ============================================================
    # CONTINUE
    # ============================================================

    return {
        "messages": [
            AIMessage(
                content=(
                    f"Продовжуємо виконання плану. "
                    f"{decision.reasoning}"
                )
            )
        ]
    }


# ================================================================
# ROUTER ПІСЛЯ REPLANNER
# ================================================================

def should_end(
    state: PlanExecuteState,
) -> Literal[
    "executor",
    "__end__",
]:
    """Продовжує виконання або завершує graph."""

    if state.get(
        "completed",
        False,
    ):

        return "__end__"


    return "executor"


# ================================================================
# LANGGRAPH
# ================================================================

graph = StateGraph(
    PlanExecuteState
)


# Nodes
graph.add_node(
    "planner",
    planner_node,
)

graph.add_node(
    "executor",
    executor_node,
)

graph.add_node(
    "checkpoint_pause",
    checkpoint_pause_node,
)

graph.add_node(
    "approval",
    approval_node,
)

graph.add_node(
    "replanner",
    replanner_node,
)


# START → planner
graph.add_edge(
    START,
    "planner",
)


# planner → executor
graph.add_edge(
    "planner",
    "executor",
)


# executor →
# approval / checkpoint_pause / replanner
graph.add_conditional_edges(
    "executor",
    route_after_executor,
)


# checkpoint_pause → replanner
graph.add_edge(
    "checkpoint_pause",
    "replanner",
)


# approval → replanner
graph.add_edge(
    "approval",
    "replanner",
)


# replanner → executor / END
graph.add_conditional_edges(
    "replanner",
    should_end,
)


# ================================================================
# SQLITE CHECKPOINTER
# ================================================================

# ФАЙЛ, а не :memory:
# Це забезпечує persistence між Python processes.
connection = sqlite3.connect(
    "agent_state.db",
    check_same_thread=False,
)


saver = SqliteSaver(
    connection
)


# interrupt_before=["approval"]: граф зупиняється ПЕРЕД виконанням
# ризикового tool — без явного виклику interrupt() у самому node.
# Це саме той механізм HITL, який вимагає завдання (item 7).
app = graph.compile(
    checkpointer=saver,
    interrupt_before=["approval"],
)


# ================================================================
# Initial State
# ================================================================

def create_initial_state(
    query: str,
    pause_after_first_step: bool = False,
) -> PlanExecuteState:
    """Створює початковий State для нового thread."""

    return {
        "messages": [
            HumanMessage(
                content=query
            )
        ],

        "plan": [],

        "current_step": 0,

        "results": [],

        "completed": False,

        "goal": "",

        "pending_tool_call": None,

        "human_decision": None,

        "pause_after_first_step": (
            pause_after_first_step
        ),

        "pause_done": False,
    }


# ================================================================
# Config helper
# ================================================================

def make_config(
    thread_id: str,
) -> dict:
    """Створює config з thread_id."""

    return {
        "configurable": {
            "thread_id": thread_id
        }
    }


# ================================================================
# Виведення interrupt
# ================================================================

def print_interrupts(
    result: dict,
) -> None:
    """Показує interrupt payload, якщо graph призупинено."""

    interrupts = result.get(
        "__interrupt__",
        []
    )


    if not interrupts:
        return


    print("\n" + "=" * 70)
    print("GRAPH INTERRUPTED")
    print("=" * 70)


    for item in interrupts:

        value = getattr(
            item,
            "value",
            item,
        )

        print(
            value
        )


def print_pending_approval(
    config: dict,
) -> bool:
    """Показує деталі ризикової дії, якщо граф зупинено interrupt_before.

    Повертає True, якщо граф справді очікує на approval-рішення людини.
    """

    snapshot = app.get_state(
        config
    )

    if snapshot.next != ("approval",):
        return False

    pending = snapshot.values.get(
        "pending_tool_call"
    )

    print("\n" + "=" * 70)
    print("GRAPH INTERRUPTED (interrupt_before='approval')")
    print("=" * 70)

    print(
        f"Tool: {pending['name']}"
    )

    print(
        f"Args: {pending['args']}"
    )

    print(
        "\nПотрібне підтвердження ризикової дії: "
        "approve / reject / edit."
    )

    return True


# ================================================================
# TASK 1 — DEMO PLAN-AND-EXECUTE
# ================================================================

def run_example(
    title: str,
    query: str,
    thread_id: str,
) -> None:
    """Запускає повний Plan-and-Execute сценарій."""

    print("\n\n" + "#" * 80)
    print(title)
    print("#" * 80)

    print(
        f"USER: {query}"
    )


    result = app.invoke(
        create_initial_state(
            query=query,
            pause_after_first_step=False,
        ),

        config=make_config(
            thread_id
        ),
    )


    print("\n" + "=" * 70)
    print("FINAL STATE")
    print("=" * 70)


    print(
        f"Completed: "
        f"{result.get('completed')}"
    )

    print(
        f"Current step: "
        f"{result.get('current_step')}"
    )


    print("\nResults:")

    for item in result.get(
        "results",
        [],
    ):

        print(
            f"- {item}"
        )


# ================================================================
# TASK 2 — CHECKPOINT DEMO: START
# ================================================================

CHECKPOINT_THREAD_ID = (
    "checkpoint-session-001"
)


def start_checkpoint_demo() -> None:
    """Запускає workflow і зупиняє його після першого кроку."""

    query = (
        "Допоможи спланувати подорож для двох людей "
        "на 7 днів. "
        "Відстань — 1200 км, "
        "головний пріоритет — швидкість. "
        "Щоденний бюджет — 75 євро на людину. "
        "Також потрібен один номер на 6 ночей "
        "по 110 євро за ніч. "
        "Порекомендуй транспорт, "
        "порахуй бюджет подорожі "
        "та вартість готелю."
    )


    config = make_config(
        CHECKPOINT_THREAD_ID
    )


    result = app.invoke(
        create_initial_state(
            query=query,
            pause_after_first_step=True,
        ),

        config=config,
    )


    print_interrupts(
        result
    )


    snapshot = app.get_state(
        config
    )


    print("\n" + "=" * 70)
    print("STATE SAVED TO SQLITE")
    print("=" * 70)


    print(
        f"Thread ID: "
        f"{CHECKPOINT_THREAD_ID}"
    )

    print(
        f"Current step: "
        f"{snapshot.values.get('current_step')}"
    )

    print(
        f"Plan: "
        f"{snapshot.values.get('plan')}"
    )

    print(
        f"Results so far: "
        f"{snapshot.values.get('results')}"
    )


    print(
        "\nТепер завершіть цей Python process "
        "і виконайте:"
    )

    print(
        "python plan_execute.py resume"
    )


# ================================================================
# TASK 2 — CHECKPOINT DEMO: RESUME
# ================================================================

def resume_checkpoint_demo() -> None:
    """Відновлює той самий thread після restart Python process."""

    config = make_config(
        CHECKPOINT_THREAD_ID
    )


    # ------------------------------------------------------------
    # Читаємо persisted state ДО resume
    # ------------------------------------------------------------

    snapshot = app.get_state(
        config
    )


    if not snapshot.values:

        print(
            "Checkpoint не знайдено. "
            "Спочатку виконайте:"
        )

        print(
            "python plan_execute.py start"
        )

        return


    print("\n" + "=" * 70)
    print("RESTORED STATE")
    print("=" * 70)


    print(
        f"Thread ID: "
        f"{CHECKPOINT_THREAD_ID}"
    )

    print(
        f"Restored current_step: "
        f"{snapshot.values.get('current_step')}"
    )

    print(
        f"Restored plan: "
        f"{snapshot.values.get('plan')}"
    )

    print(
        f"Restored results: "
        f"{snapshot.values.get('results')}"
    )


    # ------------------------------------------------------------
    # Продовжуємо той самий interrupt
    # ------------------------------------------------------------

    result = app.invoke(
        Command(
            resume={
                "action": "continue"
            }
        ),

        config=config,
    )


    print_interrupts(
        result
    )


    print("\n" + "=" * 70)
    print("RESULT AFTER RESUME")
    print("=" * 70)


    print(
        f"Completed: "
        f"{result.get('completed')}"
    )

    print(
        f"Current step: "
        f"{result.get('current_step')}"
    )


    print("\nResults:")

    for item in result.get(
        "results",
        [],
    ):

        print(
            f"- {item}"
        )


# ================================================================
# TASK 2 — INDEPENDENT THREADS
# ================================================================

def demonstrate_independent_threads() -> None:
    """Показує, що різні thread_id мають незалежний State."""

    first_config = make_config(
        "checkpoint-session-001"
    )

    second_config = make_config(
        "checkpoint-session-002"
    )


    first_state = app.get_state(
        first_config
    )

    second_state = app.get_state(
        second_config
    )


    print("\n" + "=" * 70)
    print("THREAD INDEPENDENCE")
    print("=" * 70)


    print(
        "\nthread_id = checkpoint-session-001"
    )

    print(
        f"State: {first_state.values}"
    )


    print(
        "\nthread_id = checkpoint-session-002"
    )

    print(
        f"State: {second_state.values}"
    )


    print(
        "\nРізні thread_id мають "
        "незалежні checkpoints."
    )


# ================================================================
# TASK 3 — AGENTIC RAG DEMO
# ================================================================

def run_rag_demo() -> None:
    """Демонструє, що агент сам вирішує, коли потрібна ChromaDB."""

    # ------------------------------------------------------------
    # Випадок 1:
    # RAG НЕ потрібний
    # ------------------------------------------------------------

    run_example(
        title=(
            "AGENTIC RAG DEMO 1 — "
            "search_knowledge НЕ потрібний"
        ),

        query=(
            "Я їду удвох на 5 днів. "
            "Щоденний бюджет — 80 євро на людину. "
            "Порахуй загальний бюджет."
        ),

        thread_id=(
            "rag-no-search-001"
        ),
    )


    # ------------------------------------------------------------
    # Випадок 2:
    # RAG потрібний
    # ------------------------------------------------------------

    run_example(
        title=(
            "AGENTIC RAG DEMO 2 — "
            "search_knowledge потрібний"
        ),

        query=(
            "Що потрібно перевірити "
            "перед міжнародною подорожжю? "
            "Використай внутрішню базу знань, "
            "якщо вона потрібна."
        ),

        thread_id=(
            "rag-search-001"
        ),
    )


    # ------------------------------------------------------------
    # Випадок 3:
    # Звичайний tool + RAG
    # ------------------------------------------------------------

    run_example(
        title=(
            "AGENTIC RAG DEMO 3 — "
            "travel tool + knowledge base"
        ),

        query=(
            "Я їду удвох на 5 днів. "
            "Щоденний бюджет — 80 євро на людину. "
            "Порахуй бюджет і скажи, "
            "що треба перевірити "
            "перед міжнародною поїздкою."
        ),

        thread_id=(
            "rag-combined-001"
        ),
    )


# ================================================================
# TASK 4 — HITL START
# ================================================================

DEFAULT_HITL_THREAD_ID = (
    "hitl-demo-001"
)


def start_hitl_demo(
    thread_id: str,
) -> None:
    """Запускає сценарій до interrupt_before='approval' перед book_hotel."""

    query = (
        "Я хочу забронювати Demo Travel Hotel "
        "з 2026-09-15 на 4 ночі. "
        "Загальна вартість — 400 євро. "
        "Виконай бронювання."
    )


    config = make_config(
        thread_id
    )


    app.invoke(
        create_initial_state(
            query=query,
            pause_after_first_step=False,
        ),

        config=config,
    )


    print_pending_approval(
        config
    )


    print(
        f"\nHITL thread_id: {thread_id}"
    )

    print(
        "\nДля продовження використайте "
        "ТОЙ САМИЙ thread_id:"
    )

    print(
        f"python plan_execute.py approve {thread_id}"
    )

    print(
        f"python plan_execute.py reject {thread_id}"
    )

    print(
        f"python plan_execute.py edit {thread_id}"
    )


# ================================================================
# TASK 4 — HITL APPROVE
# ================================================================

def approve_hitl(
    thread_id: str,
) -> None:
    """Підтверджує ризиковий tool через update_state() + invoke(None, ...)."""

    config = make_config(
        thread_id
    )

    # Записуємо рішення людини у state ДО відновлення graph.
    app.update_state(
        config,
        {
            "human_decision": {
                "action": "approve"
            }
        },
    )

    # None означає "продовжити з місця зупинки interrupt_before".
    result = app.invoke(
        None,
        config=config,
    )


    print("\n" + "=" * 70)
    print("APPROVE RESULT")
    print("=" * 70)


    for item in result.get(
        "results",
        [],
    ):

        print(
            f"- {item}"
        )


# ================================================================
# TASK 4 — HITL REJECT
# ================================================================

def reject_hitl(
    thread_id: str,
) -> None:
    """Відхиляє ризиковий tool через update_state() + invoke(None, ...)."""

    config = make_config(
        thread_id
    )

    app.update_state(
        config,
        {
            "human_decision": {
                "action": "reject",

                "reason": (
                    "Користувач вирішив "
                    "не виконувати бронювання."
                ),
            }
        },
    )

    result = app.invoke(
        None,
        config=config,
    )


    print("\n" + "=" * 70)
    print("REJECT RESULT")
    print("=" * 70)


    for item in result.get(
        "results",
        [],
    ):

        print(
            f"- {item}"
        )


# ================================================================
# TASK 4 — HITL EDIT
# ================================================================

def edit_hitl(
    thread_id: str,
) -> None:
    """Змінює параметри risky tool і виконує його через update_state()."""

    config = make_config(
        thread_id
    )

    app.update_state(
        config,
        {
            "human_decision": {
                "action": "edit",

                "args": {
                    "hotel_name": (
                        "Demo Travel Hotel"
                    ),

                    "check_in": (
                        "2026-09-16"
                    ),

                    "nights": 3,

                    "total_cost": 300,
                },
            }
        },
    )

    result = app.invoke(
        None,
        config=config,
    )


    print("\n" + "=" * 70)
    print("EDIT RESULT")
    print("=" * 70)


    for item in result.get(
        "results",
        [],
    ):

        print(
            f"- {item}"
        )


# ================================================================
# TASK 1 — 3 PLAN-AND-EXECUTE EXAMPLES
# ================================================================

def run_plan_execute_examples() -> None:
    """Демонструє Завдання 1 на трьох рівнях складності."""

    # SIMPLE
    run_example(
        title="EXAMPLE 1 — SIMPLE",

        query=(
            "Я їду удвох на 5 днів. "
            "Щоденний бюджет на одну людину — 80 євро. "
            "Порахуй загальний бюджет подорожі."
        ),

        thread_id="plan-simple-001",
    )


    # MEDIUM
    run_example(
        title="EXAMPLE 2 — MEDIUM",

        query=(
            "Я планую подорож для двох людей на 4 дні. "
            "Щоденний бюджет — 70 євро на людину. "
            "Також потрібен один номер у готелі "
            "на 4 ночі по 100 євро. "
            "Порахуй бюджет подорожі "
            "та вартість готелю."
        ),

        thread_id="plan-medium-001",
    )


    # COMPLEX
    run_example(
        title="EXAMPLE 3 — COMPLEX",

        query=(
            "Допоможи спланувати подорож "
            "для двох людей на 7 днів. "
            "Відстань — 1200 км, "
            "головний пріоритет — швидкість. "
            "Щоденний бюджет — 75 євро на людину. "
            "Також потрібен один номер "
            "на 6 ночей по 110 євро. "
            "Порекомендуй транспорт, "
            "порахуй бюджет та вартість проживання."
        ),

        thread_id="plan-complex-001",
    )


# ================================================================
# ВІЗУАЛІЗАЦІЯ (додаткова вимога): Mermaid-діаграма графа
# ================================================================

def export_graph_diagram() -> str:
    """Зберігає Mermaid-діаграму Plan-and-Execute графа у graphs/plan_execute.mmd."""

    os.makedirs("graphs", exist_ok=True)

    mermaid_text = app.get_graph().draw_mermaid()

    path = os.path.join("graphs", "plan_execute.mmd")

    with open(path, "w", encoding="utf-8") as f:
        f.write(mermaid_text)

    print(mermaid_text)
    print(f"\nMermaid-діаграма збережена у: {path}")

    return path


# ================================================================
# CLI
# ================================================================

def print_help() -> None:
    """Показує доступні команди."""

    print(
        """
============================================================
HW2 — Plan-and-Execute Agent
============================================================

ЗАВДАННЯ 1 — PLAN-AND-EXECUTE

python plan_execute.py simple
    Запустити simple-приклад.

python plan_execute.py medium
    Запустити medium-приклад.

python plan_execute.py complex
    Запустити complex-приклад.

python plan_execute.py demo
    Запустити всі 3 приклади:
    simple / medium / complex.


ЗАВДАННЯ 2 — CHECKPOINTER

python plan_execute.py start
    Запустити workflow та призупинити
    після першого кроку.

python plan_execute.py resume
    У новому Python process відновити
    той самий state з agent_state.db.

python plan_execute.py threads
    Показати незалежність різних thread_id.


ЗАВДАННЯ 3 — AGENTIC RAG

python plan_execute.py rag
    Запустити приклади Agentic RAG.


ЗАВДАННЯ 4 — HITL

python plan_execute.py hitl hitl-approve-001
python plan_execute.py approve hitl-approve-001

python plan_execute.py hitl hitl-reject-001
python plan_execute.py reject hitl-reject-001

python plan_execute.py hitl hitl-edit-001
python plan_execute.py edit hitl-edit-001


ВІЗУАЛІЗАЦІЯ

python plan_execute.py graph
    Вивести та зберегти Mermaid-діаграму графа
    (graphs/plan_execute.mmd).

============================================================
"""
    )


if __name__ == "__main__":

    command = (
        sys.argv[1].lower()
        if len(sys.argv) > 1
        else "help"
    )

    thread_id = (
        sys.argv[2]
        if len(sys.argv) > 2
        else DEFAULT_HITL_THREAD_ID
    )


    if command == "simple":

        run_example(
            title="EXAMPLE 1 — SIMPLE",

            query=(
                "Я їду удвох на 5 днів. "
                "Щоденний бюджет на одну людину — 80 євро. "
                "Порахуй загальний бюджет подорожі."
            ),

            thread_id="plan-simple-001",
        )


    elif command == "medium":

        run_example(
            title="EXAMPLE 2 — MEDIUM",

            query=(
                "Я планую подорож для двох людей на 4 дні. "
                "Щоденний бюджет — 70 євро на людину. "
                "Також потрібен один номер у готелі "
                "на 4 ночі по 100 євро. "
                "Порахуй бюджет подорожі "
                "та вартість готелю."
            ),

            thread_id="plan-medium-001",
        )


    elif command == "complex":

        run_example(
            title="EXAMPLE 3 — COMPLEX",

            query=(
                "Допоможи спланувати подорож "
                "для двох людей на 7 днів. "
                "Відстань — 1200 км, "
                "головний пріоритет — швидкість. "
                "Щоденний бюджет — 75 євро на людину. "
                "Також потрібен один номер "
                "на 6 ночей по 110 євро. "
                "Порекомендуй транспорт, "
                "порахуй бюджет та вартість проживання."
            ),

            thread_id="plan-complex-001",
        )


    elif command == "demo":

        run_plan_execute_examples()


    elif command == "start":

        start_checkpoint_demo()


    elif command == "resume":

        resume_checkpoint_demo()


    elif command == "threads":

        demonstrate_independent_threads()


    elif command == "rag":

        run_rag_demo()


    elif command == "hitl":

        start_hitl_demo(
            thread_id
        )


    elif command == "approve":

        approve_hitl(
            thread_id
        )


    elif command == "reject":

        reject_hitl(
            thread_id
        )


    elif command == "edit":

        edit_hitl(
            thread_id
        )


    elif command == "graph":

        export_graph_diagram()


    else:

        print_help()