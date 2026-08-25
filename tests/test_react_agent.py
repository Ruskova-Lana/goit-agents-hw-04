"""Тести ReAct-агента: чисті guardrail-функції + базовий ReAct-цикл.

Реальний Gemini LLM у тестах НЕ використовується — замість нього
підставляється фейкова chat-модель (FakeToolCallingModel), яка
повертає заздалегідь задану послідовність AIMessage. Це дозволяє
детерміновано перевірити tool-calling цикл та guardrails без
мережевих викликів чи API-ключа.
"""

from langchain_core.messages import AIMessage, HumanMessage

from react_agent import (
    MAX_STEPS,
    build_app,
    create_initial_state,
    is_repeated_call,
    make_call_signature,
)


# ================================================================
# Fake chat model
# ================================================================

class FakeToolCallingModel:
    """Мінімальна заглушка chat-моделі: bind_tools() + invoke()."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        response = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        return response


def _tool_call_message(name: str, args: dict, call_id: str = "call_1") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id}],
    )


def _final_message(text: str) -> AIMessage:
    return AIMessage(content=text, tool_calls=[])


# ================================================================
# Чисті guardrail-функції
# ================================================================

def test_make_call_signature_is_order_independent():
    sig_a = make_call_signature("calculate_trip_budget", {"a": 1, "b": 2})
    sig_b = make_call_signature("calculate_trip_budget", {"b": 2, "a": 1})
    assert sig_a == sig_b


def test_is_repeated_call_detects_duplicate():
    history = [make_call_signature("calculate_trip_budget", {"days": 5})]
    signature = make_call_signature("calculate_trip_budget", {"days": 5})
    assert is_repeated_call(history, signature) is True


def test_is_repeated_call_allows_new_call():
    history = [make_call_signature("calculate_trip_budget", {"days": 5})]
    signature = make_call_signature("calculate_trip_budget", {"days": 6})
    assert is_repeated_call(history, signature) is False


# ================================================================
# Базовий ReAct-цикл: LLM -> tool -> LLM -> фінальна відповідь
# ================================================================

def test_react_loop_calls_tool_then_finishes():
    fake_llm = FakeToolCallingModel(
        [
            _tool_call_message(
                "calculate_trip_budget",
                {"travelers": 2, "days": 5, "daily_budget": 80},
            ),
            _final_message("Загальний бюджет становить €800.00."),
        ]
    )

    app = build_app(chat_model=fake_llm)

    result = app.invoke(
        create_initial_state("Порахуй бюджет подорожі."),
        config={"recursion_limit": 50},
    )

    final_message = result["messages"][-1]

    assert isinstance(final_message, AIMessage)
    assert "800" in final_message.content
    assert fake_llm.calls == 2

    # Траєкторія має містити і виклик LLM, і виконання tool
    events = [entry["event"] for entry in result["trajectory"]]
    assert "llm_call" in events
    assert "tool_call" in events


def test_react_loop_terminates_on_repeated_calls_without_infinite_loop():
    # LLM щоразу викликає той самий tool -> repeat guardrail має
    # примусово завершити граф задовго до нескінченного циклу.
    looping_response = _tool_call_message(
        "calculate_trip_budget",
        {"travelers": 2, "days": 5, "daily_budget": 80},
    )

    fake_llm = FakeToolCallingModel([looping_response])

    app = build_app(chat_model=fake_llm)

    result = app.invoke(
        create_initial_state("Порахуй бюджет кілька разів."),
        config={"recursion_limit": 100},
    )

    assert fake_llm.calls <= MAX_STEPS + 1

    final_message = result["messages"][-1]
    assert isinstance(final_message, AIMessage)


class VaryingArgsToolCallingModel:
    """Фейкова модель, що щоразу викликає tool з НОВИМИ аргументами,
    аби repeat-detection жодного разу не спрацював і межу кроків
    перевіряв саме guardrail max_steps."""

    def __init__(self):
        self.calls = 0

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.calls += 1
        return _tool_call_message(
            "calculate_trip_budget",
            {"travelers": 2, "days": self.calls, "daily_budget": 80},
            call_id=f"call_{self.calls}",
        )


def test_react_loop_respects_max_steps_without_repeats():
    fake_llm = VaryingArgsToolCallingModel()

    app = build_app(chat_model=fake_llm)

    result = app.invoke(
        create_initial_state("Постійно рахуй бюджет з новими даними."),
        config={"recursion_limit": 100},
    )

    # agent_node перестає викликати LLM, щойно steps >= MAX_STEPS
    assert fake_llm.calls == MAX_STEPS

    events = [entry["event"] for entry in result["trajectory"]]
    assert "max_steps_exceeded" in events
    assert "repeated_call_blocked" not in events


def test_react_loop_blocks_repeated_tool_call():
    fake_llm = FakeToolCallingModel(
        [
            _tool_call_message(
                "calculate_trip_budget",
                {"travelers": 2, "days": 5, "daily_budget": 80},
            ),
            _tool_call_message(
                "calculate_trip_budget",
                {"travelers": 2, "days": 5, "daily_budget": 80},
            ),
            _final_message("Готово."),
        ]
    )

    app = build_app(chat_model=fake_llm)

    result = app.invoke(
        create_initial_state("Порахуй бюджет."),
        config={"recursion_limit": 50},
    )

    events = [entry["event"] for entry in result["trajectory"]]
    assert "repeated_call_blocked" in events


def test_initial_state_has_zero_steps():
    state = create_initial_state("тестовий запит")
    assert state["steps"] == 0
    assert state["repeat_count"] == 0
    assert isinstance(state["messages"][0], HumanMessage)
