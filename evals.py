"""Scenario-based evals для MAS (ДЗ4 Завд. 5) — розширення tests/test.json з ДЗ1.

ДЗ1 тестував ОДИН агент (ReAct). Тут — увесь MAS: supervisor routing,
кілька агентів, RAG, HITL. 5 сценаріїв, що покривають:

- EVAL-01 simple billing        (один агент, один tool)
- EVAL-02 multi-step tech       (один агент, кілька викликів tool)
- EVAL-03 RAG-heavy             (researcher, Agentic RAG)
- EVAL-04 cross-agent           (запит, що потребує 2 доменів одразу)
- EVAL-05 HITL flow             (billing → book_hotel → interrupt() → approve)

Оригінальні тексти сценаріїв у завданні написані для generic
support-ticket домену (get_ticket/update_ticket_status), якого в цьому
проєкті немає — адаптовано до реального домену цього MAS (подорожі:
calculate_trip_budget/estimate_hotel_cost/recommend_transport/
search_knowledge/book_hotel), зберігши ТИП кожного сценарію.

Результат — eval_results.json, поля: scenario_id, type, query,
expected, status (pass/fail/partial), latency_ms, agents_used,
tools_called, notes.

Запуск:
    python evals.py
"""

import json
import sys
import time
import uuid

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from datetime import datetime, timezone

import mas_langgraph as mas


RESULTS_PATH = "eval_results.json"

# mas_state.db персистентний per thread_id — унікальний suffix на кожен
# запуск запобігає накопиченню messages/trajectory з попередніх прогонів
# (Annotated[list, operator.add]) при повторному python evals.py.
RUN_ID = uuid.uuid4().hex[:8]


def _run(query: str, thread_id: str, pause_after_first_step: bool = False):
    start = time.monotonic()
    result = mas.app.invoke(
        mas.create_initial_state(query, pause_after_first_step=pause_after_first_step),
        config=mas.make_config(thread_id),
    )
    elapsed_ms = round((time.monotonic() - start) * 1000)
    return result, elapsed_ms


def _agents_used(result: dict) -> list[str]:
    agents = {
        e["agent_name"]
        for e in result.get("trajectory", [])
        if e.get("agent_name") not in (None, "supervisor")
    }
    return sorted(agents)


def _tools_called(result: dict) -> list[str]:
    tools = []
    for e in result.get("trajectory", []):
        if e.get("event") == "tool_call" and "tool" in e:
            tools.append(e["tool"])
        elif e.get("event") == "hitl_decision" and e.get("action") in ("approve", "edit"):
            # billing_approval_node логує виконаний ризиковий tool під
            # окремою подією "hitl_decision" (не "tool_call") — reject
            # НЕ рахується як виклик, бо tool фактично не виконувався.
            tools.append(e["tool"])
    return tools


def _final_text(result: dict) -> str:
    if result.get("results"):
        return " | ".join(result["results"])
    last_message = result["messages"][-1]
    return mas._extract_text(last_message.content)


# ================================================================
# EVAL-01 — Simple billing (один агент, один tool)
# ================================================================

def eval_01() -> dict:
    query = "Я їду сама на 3 дні. Щоденний бюджет — 60 євро. Порахуй загальний бюджет подорожі."
    result, elapsed_ms = _run(query, f"eval-01-simple-billing-{RUN_ID}")

    agents_used = _agents_used(result)
    tools_called = _tools_called(result)

    checks = {
        "routed_to_billing": agents_used == ["billing"],
        "called_calculate_trip_budget": "calculate_trip_budget" in tools_called,
        "no_other_tools": all(t == "calculate_trip_budget" for t in tools_called),
        "single_tool_call": len(tools_called) == 1,
    }
    status = "pass" if all(checks.values()) else "fail"

    return {
        "scenario_id": "EVAL-01",
        "type": "simple_billing",
        "query": query,
        "expected": "supervisor -> billing; єдиний tool_call: calculate_trip_budget",
        "status": status,
        "latency_ms": elapsed_ms,
        "agents_used": agents_used,
        "tools_called": tools_called,
        "answer": _final_text(result),
        "checks": checks,
    }


# ================================================================
# EVAL-02 — Multi-step tech (один агент, кілька tool calls)
# ================================================================

def eval_02() -> dict:
    query = (
        "Порекомендуй транспорт для двох окремих маршрутів: перший — 300 км, "
        "пріоритет дешевше; другий — 1200 км, пріоритет швидкість."
    )
    result, elapsed_ms = _run(query, f"eval-02-multistep-tech-{RUN_ID}")

    agents_used = _agents_used(result)
    tools_called = _tools_called(result)
    transport_calls = [t for t in tools_called if t == "recommend_transport"]

    checks = {
        "routed_to_tech": agents_used == ["tech"],
        "used_recommend_transport": "recommend_transport" in tools_called,
        "no_other_tools": all(t == "recommend_transport" for t in tools_called),
    }
    # "Multi-step" в оцінному сенсі: 2 окремі виклики recommend_transport
    # (по одному на маршрут). ReAct-агент з ОДНИМ tool може легітимно
    # вирішити обробити обидва маршрути одним викликом або двома —
    # фіксуємо РЕАЛЬНУ поведінку, а не форсуємо очікування.
    multi_call = len(transport_calls) >= 2
    status = "pass" if all(checks.values()) and multi_call else ("partial" if all(checks.values()) else "fail")
    note = (
        f"LLM викликав recommend_transport {len(transport_calls)} раз(и) "
        f"({'multi-step підтверджено' if multi_call else 'модель обробила обидва маршрути за 1 виклик — architecturally OK, але не multi-step'})."
    )

    return {
        "scenario_id": "EVAL-02",
        "type": "multi_step_tech",
        "query": query,
        "expected": "supervisor -> tech; 2-3 кроки (>=2 виклики recommend_transport)",
        "status": status,
        "latency_ms": elapsed_ms,
        "agents_used": agents_used,
        "tools_called": tools_called,
        "answer": _final_text(result),
        "checks": checks,
        "notes": note,
    }


# ================================================================
# EVAL-03 — RAG-heavy (researcher, Agentic RAG)
# ================================================================

def eval_03() -> dict:
    query = "Які документи потрібні перед міжнародною подорожжю і чи варто оформити страхування?"
    result, elapsed_ms = _run(query, f"eval-03-rag-heavy-{RUN_ID}")

    agents_used = _agents_used(result)
    tools_called = _tools_called(result)
    answer = _final_text(result)

    checks = {
        "routed_to_researcher": agents_used == ["researcher"],
        "used_search_knowledge": "search_knowledge" in tools_called,
        "no_other_tools": all(t == "search_knowledge" for t in tools_called),
        "mentions_insurance_or_documents": any(
            kw in answer.lower() for kw in ("страхуван", "паспорт", "документ")
        ),
    }
    status = "pass" if all(checks.values()) else "fail"

    return {
        "scenario_id": "EVAL-03",
        "type": "rag_heavy",
        "query": query,
        "expected": "supervisor -> researcher; tool: search_knowledge (Agentic RAG, ChromaDB); "
        "довідково — той самий контент доступний і як MCP resource travel://knowledge-base",
        "status": status,
        "latency_ms": elapsed_ms,
        "agents_used": agents_used,
        "tools_called": tools_called,
        "answer": answer,
        "checks": checks,
    }


# ================================================================
# EVAL-04 — Cross-agent (запит потребує 2 доменів одночасно)
# ================================================================

def eval_04() -> dict:
    query = (
        "Порекомендуй транспорт на 900 км (пріоритет швидкість) і порахуй бюджет "
        "подорожі для 2 осіб на 6 днів по 85 євро на день."
    )
    result, elapsed_ms = _run(query, f"eval-04-cross-agent-{RUN_ID}")

    agents_used = _agents_used(result)
    tools_called = _tools_called(result)

    used_transport = "recommend_transport" in tools_called
    used_budget = "calculate_trip_budget" in tools_called
    both_covered = used_transport and used_budget

    # MAS НЕ має явного agent-to-agent handoff (supervisor обирає РІВНО
    # одного агента). Єдиний спосіб покрити крос-доменний запит — якщо
    # supervisor класифікує його як "general" (катч-ол з усіма tools).
    if both_covered:
        status = "pass"
        note = (
            f"supervisor маршрутизував на {agents_used} — обидва tools викликано "
            "в межах ОДНОГО агента (general має повний набір tools), тому "
            "результат покриває обидва домени навіть без явного handoff."
        )
    elif len(agents_used) == 1 and (used_transport or used_budget):
        status = "fail"
        note = (
            f"supervisor маршрутизував лише на {agents_used}, який не має tool для "
            "другого домену запиту (немає agent-to-agent handoff у поточній "
            "архітектурі MAS) — ВІДОМЕ АРХІТЕКТУРНЕ ОБМЕЖЕННЯ: одноходовий "
            "supervisor, немає ланцюжка агентів у межах одного запиту."
        )
    else:
        status = "fail"
        note = "Жоден з очікуваних tools не викликано."

    return {
        "scenario_id": "EVAL-04",
        "type": "cross_agent",
        "query": query,
        "expected": "supervisor -> billing АБО tech (з handoff) — або general, якщо LLM "
        "класифікує запит як змішаний",
        "status": status,
        "latency_ms": elapsed_ms,
        "agents_used": agents_used,
        "tools_called": tools_called,
        "answer": _final_text(result),
        "checks": {"used_recommend_transport": used_transport, "used_calculate_trip_budget": used_budget},
        "notes": note,
    }


# ================================================================
# EVAL-05 — HITL flow (billing -> book_hotel -> interrupt() -> approve)
# ================================================================

def eval_05() -> dict:
    query = (
        "Забронюй, будь ласка, Demo Travel Hotel з 2026-10-01 на 3 ночі за 300 євро. "
        "Я підтверджую бронювання."
    )
    thread_id = f"eval-05-hitl-flow-{RUN_ID}"
    result, elapsed_ms_1 = _run(query, thread_id)

    # billing_approval_node НЕ викликає interrupt() всередині — граф
    # зупиняється СТАТИЧНО через interrupt_before=["billing_approval"],
    # тому result["__interrupt__"] тут ПОРОЖНІЙ (це поле заповнюється
    # лише для динамічного interrupt() всередині вузла, як у
    # approval_gate/billing_pause). Правильна перевірка паузи —
    # pending_tool_call у самому стані (повертається з invoke() навіть
    # при зупинці) + snapshot.next з checkpointer.
    config = mas.make_config(thread_id)
    pending = result.get("pending_tool_call")
    snapshot_next = mas.app.get_state(config).next

    # рішення передається через app.update_state() + invoke(None, ...),
    # НЕ через Command(resume=...) (той працює лише для approval_gate,
    # де interrupt() викликається динамічно всередині вузла).
    start = time.monotonic()
    mas.app.update_state(config, {"human_decision": {"action": "approve"}})
    resumed = mas.app.invoke(None, config=config)
    elapsed_ms_2 = round((time.monotonic() - start) * 1000)

    agents_used = _agents_used(resumed)
    tools_called = _tools_called(resumed)
    booked = any("booking_id" in r for r in resumed.get("results", []))

    checks = {
        "routed_to_billing": "billing" in agents_used,
        "graph_paused_before_book_hotel": snapshot_next == ("billing_approval",)
        and bool(pending)
        and pending.get("name") == "book_hotel",
        "book_hotel_called_after_approve": "book_hotel" in tools_called,
        "booking_confirmed": booked,
    }
    status = "pass" if all(checks.values()) else "fail"

    return {
        "scenario_id": "EVAL-05",
        "type": "hitl_flow",
        "query": query,
        "expected": "billing -> book_hotel -> graph pauses (interrupt_before='billing_approval') -> "
        "app.update_state({'human_decision': {'action': 'approve'}}) + invoke(None) -> booking_id",
        "status": status,
        "latency_ms": elapsed_ms_1 + elapsed_ms_2,
        "agents_used": agents_used,
        "tools_called": tools_called,
        "answer": _final_text(resumed),
        "checks": checks,
        "pending_tool_call_at_pause": pending,
    }


EVALS = [eval_01, eval_02, eval_03, eval_04, eval_05]


def run_all() -> None:
    results = []
    for eval_fn in EVALS:
        print(f"\n{'=' * 70}\nRunning {eval_fn.__name__}...\n{'=' * 70}")
        result = eval_fn()
        print(f"[{result['scenario_id']}] {result['type']} -> {result['status'].upper()}")
        print(f"  agents_used={result['agents_used']} tools_called={result['tools_called']}")
        print(f"  latency_ms={result['latency_ms']}")
        if result.get("notes"):
            print(f"  notes: {result['notes']}")
        results.append(result)

    summary = {
        "total_scenarios": len(results),
        "passed": sum(1 for r in results if r["status"] == "pass"),
        "partial": sum(1 for r in results if r["status"] == "partial"),
        "failed": sum(1 for r in results if r["status"] == "fail"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    payload = {"summary": summary, "results": results}
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nЗбережено у: {RESULTS_PATH}")


if __name__ == "__main__":
    run_all()
