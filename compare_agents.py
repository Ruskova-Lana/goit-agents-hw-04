"""Числове порівняння ReAct vs Plan-and-Execute на ОДНІЙ і тій самій задачі.

Обидва агенти отримують ідентичний запит (без book_hotel — цей ризиковий
tool підключений лише до Plan-and-Execute, тому виключений із чесного
порівняння) і порівнюються за: часом виконання, кількістю LLM-викликів
та кількістю tool-викликів.
"""

import json
import os
import sys
import time
import uuid

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

load_dotenv()


QUERY = (
    "Я їду удвох на 5 днів. Щоденний бюджет на людину — 80 євро. "
    "Також потрібен готель на 4 ночі по 100 євро за ніч. "
    "Порахуй бюджет подорожі, вартість готелю і скажи, "
    "що перевірити перед міжнародною подорожжю."
)


def run_react() -> dict:
    """Запускає ReAct-агента на QUERY і збирає метрики з trajectory."""

    from react_agent import build_app, create_initial_state

    app = build_app()

    start = time.monotonic()
    result = app.invoke(create_initial_state(QUERY), config={"recursion_limit": 50})
    elapsed = time.monotonic() - start

    trajectory = result.get("trajectory", [])
    llm_calls = sum(1 for e in trajectory if e["event"] == "llm_call")
    tool_calls = sum(1 for e in trajectory if e["event"] == "tool_call")

    return {
        "agent": "ReAct",
        "elapsed_seconds": round(elapsed, 2),
        "llm_calls": llm_calls,
        "tool_calls": tool_calls,
        "final_answer": str(result["messages"][-1].content),
    }


def run_plan_execute() -> dict:
    """Запускає Plan-and-Execute агента на QUERY у новому thread_id."""

    import plan_execute as pe

    thread_id = f"compare-{uuid.uuid4().hex[:8]}"
    config = pe.make_config(thread_id)

    node_counts: dict[str, int] = {}

    start = time.monotonic()

    for update in pe.app.stream(
        pe.create_initial_state(query=QUERY, pause_after_first_step=False),
        config=config,
        stream_mode="updates",
    ):
        for node_name in update:
            node_counts[node_name] = node_counts.get(node_name, 0) + 1

    elapsed = time.monotonic() - start

    # По одному LLM-виклику на кожне звернення до planner/executor/replanner
    llm_calls = (
        node_counts.get("planner", 0)
        + node_counts.get("executor", 0)
        + node_counts.get("replanner", 0)
    )

    final_state = pe.app.get_state(config).values
    results = final_state.get("results", [])

    # Крок є tool-викликом, якщо результат — JSON зі "status" від safe_tool_invoke
    tool_calls = sum(1 for r in results if '"status"' in r)

    return {
        "agent": "Plan-and-Execute",
        "elapsed_seconds": round(elapsed, 2),
        "llm_calls": llm_calls,
        "tool_calls": tool_calls,
        "final_answer": " | ".join(results),
    }


def print_table(rows: list[dict]) -> None:
    print("\n" + "=" * 78)
    print("ЧИСЛОВЕ ПОРІВНЯННЯ: ReAct vs Plan-and-Execute (та сама задача)")
    print("=" * 78)

    header = f"{'Агент':<20}{'Час(с)':<10}{'LLM-викликів':<15}{'Tool-викликів':<10}"
    print(header)
    print("-" * 78)

    for row in rows:
        print(
            f"{row['agent']:<20}{row['elapsed_seconds']:<10}"
            f"{row['llm_calls']:<15}{row['tool_calls']:<10}"
        )


def main() -> None:
    print(f"Запит: {QUERY}\n")

    print("--- ReAct ---")
    react_result = run_react()
    print(f"Час: {react_result['elapsed_seconds']}с | "
          f"LLM-викликів: {react_result['llm_calls']} | "
          f"Tool-викликів: {react_result['tool_calls']}")
    print(f"Відповідь: {react_result['final_answer']}\n")

    print("--- Plan-and-Execute ---")
    pe_result = run_plan_execute()
    print(f"Час: {pe_result['elapsed_seconds']}с | "
          f"LLM-викликів: {pe_result['llm_calls']} | "
          f"Tool-викликів: {pe_result['tool_calls']}")
    print(f"Результати: {pe_result['final_answer']}\n")

    rows = [react_result, pe_result]
    print_table(rows)

    os.makedirs("logs", exist_ok=True)
    path = os.path.join("logs", "agent_comparison.json")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print(f"\nРезультати збережено у {path}")


if __name__ == "__main__":
    main()
