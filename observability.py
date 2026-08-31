"""Observability (ДЗ4 Завд. 5) — явне налаштування LangSmith-трейсингу.

На відміну від "мовчазного" покладання на те, що LangChain/LangGraph
самі підхоплять `LANGSMITH_*` змінні з `.env` (це теж працює — саме
так тут і відбувається технічно), цей модуль робить конфігурацію
ЯВНОЮ й інспектованою:

- `configure_tracing()` — читає `.env`, перевіряє наявність
  `LANGSMITH_API_KEY`, гарантує `LANGSMITH_PROJECT` за замовчуванням
  (якщо власник проєкту не задав свій), і повертає структурований
  статус (не просто True/False) — щоб `mas_langgraph.py`/`evals.py`/
  `red_team.py` могли залогувати, чи трейсинг реально активний.
- `traced_config(thread_id, **extra)` — будує LangGraph `config` з
  `configurable.thread_id` ТА `tags`/`metadata`/`run_name` для
  LangSmith — щоб трейси різних агентів/сценаріїв можна було
  фільтрувати в дашборді, а не шукати серед сотні безіменних runs.

Використання (mas_langgraph.py):
    import observability
    observability.configure_tracing()
    ...
    result = app.invoke(state, config=observability.traced_config(thread_id, agent_hint="billing"))
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import os

from dotenv import load_dotenv

load_dotenv()


DEFAULT_PROJECT = "goit-mas-travel-assistant"


def configure_tracing(project_name: str = DEFAULT_PROJECT) -> dict:
    """Явно вмикає/перевіряє LangSmith-трейсинг.

    Returns:
        {"enabled": bool, "project": str, "reason": str} — статус, який
        варто залогувати при старті, а не мовчки покладатись, що
        трейсинг "десь там" працює.
    """

    api_key = os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY")

    if not api_key:
        return {
            "enabled": False,
            "project": None,
            "reason": "LANGSMITH_API_KEY відсутній у .env — трейсинг вимкнено.",
        }

    # Гарантуємо LANGSMITH_TRACING_V2=true, навіть якщо .env забув —
    # ключ без цього прапорця не активує інструментацію.
    os.environ.setdefault("LANGSMITH_TRACING_V2", "true")
    os.environ.setdefault("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

    project = os.environ.get("LANGSMITH_PROJECT") or project_name
    os.environ["LANGSMITH_PROJECT"] = project

    return {
        "enabled": True,
        "project": project,
        "reason": f"LANGSMITH_API_KEY знайдено, LANGSMITH_TRACING_V2=true, project={project!r}.",
    }


def traced_config(thread_id: str, agent_hint: str | None = None, **extra_metadata) -> dict:
    """LangGraph config з thread_id + LangSmith tags/metadata/run_name.

    `agent_hint` (напр. "billing"/"tech"/"eval"/"red-team") потрапляє і
    в tags (для фільтрації в LangSmith UI), і в run_name (щоб список
    runs був читабельний без відкриття кожного трейсу).
    """

    tags = ["mas", "travel-assistant"]
    if agent_hint:
        tags.append(agent_hint)

    run_name = f"mas:{agent_hint}:{thread_id}" if agent_hint else f"mas:{thread_id}"

    return {
        "configurable": {"thread_id": thread_id},
        "tags": tags,
        "metadata": {"thread_id": thread_id, "agent_hint": agent_hint, **extra_metadata},
        "run_name": run_name,
    }


if __name__ == "__main__":
    status = configure_tracing()
    print(status)
    if status["enabled"]:
        print(f"\nПриклад traced_config(): {traced_config('demo-thread-001', agent_hint='billing')}")
    else:
        print("\nДодайте LANGSMITH_API_KEY у .env, щоб увімкнути трейсинг.")
