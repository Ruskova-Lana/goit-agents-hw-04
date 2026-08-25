# Туристичний AI-агент: ReAct + Plan-and-Execute

## 1. Опис проєкту

Туристичний AI-агент на базі LangGraph, реалізований у **двох архітектурах**:

* **ReAct** (`react_agent.py`) — на кожній ітерації LLM сама вирішує: викликати
  tool чи одразу відповісти. Захищений `max_steps`, `timeout` та детекцією
  повторних викликів tools.
* **Plan-and-Execute** (`plan_execute.py`) — planner спочатку будує повний план
  виконання задачі, а потім executor послідовно виконує його кроки, за потреби
  коригуючи план (**replanning**) на основі проміжних результатів.

Проєкт демонструє п'ять окремих можливостей:

1. **ReAct-агент** — цикл LLM → tools → LLM з `max_steps=10`, `timeout=120с`,
   детекцією повторних викликів tools та JSON-логуванням траєкторії.
2. **Plan-and-Execute** — planner будує план, executor виконує кроки, replanner вирішує
   `continue` / `replan` / `finish`.
3. **Checkpointer (persistence)** — стан графа Plan-and-Execute зберігається у SQLite
   (`agent_state.db`) і переживає перезапуск Python-процесу.
4. **Agentic RAG** — агент сам вирішує, коли йому потрібна довідкова інформація з
   внутрішньої бази знань (ChromaDB), а коли достатньо звичайних tools.
5. **Human-in-the-Loop (HITL)** — ризикові дії (бронювання готелю) виконуються лише після
   явного підтвердження людини: граф скомпільований з `interrupt_before=["approval"]`,
   рішення передається через `app.update_state()` + `app.invoke(None, config)`.

## 2. Архітектура

### 2.1 ReAct-агент (`react_agent.py`)

```
START → agent ─── tool_calls? ──yes──→ tools ──→ agent (цикл)
           │
           no
           ↓
          END
```

* **agent** — викликає LLM (`bind_tools`) з системним промптом; перед кожним
  зверненням до LLM перевіряються guardrails `max_steps=10` і `timeout=120с` —
  якщо перевищено, граф примусово завершується без нового виклику LLM.
* **tools** — виконує усі `tool_calls` останнього `AIMessage`. Перед виконанням
  кожен виклик звіряється із сигнатурою (`tool_name` + `args`) уже виконаних
  викликів (`call_history`): якщо це повтор — tool НЕ виконується повторно, а
  LLM отримує підказку-`ToolMessage` не повторювати виклик. Якщо повторів
  накопичилось ≥3 (`MAX_REPEATED_CALLS`) — граф примусово завершується.
* Кожен крок (виклик LLM, виклик tool, спрацювання guardrail) додається у
  список `trajectory` і наприкінці запуску зберігається як JSON-файл у
  `logs/react_<thread_id>_<timestamp>.json`.
* Ризиковий tool `book_hotel` навмисно НЕ підключений до ReAct-агента — HITL
  для нього демонструється окремо у Plan-and-Execute (розділ 8).

### 2.2 Plan-and-Execute (`plan_execute.py`)

```
START → planner → executor ─┬─→ approval ─────┐
                             ├─→ checkpoint_pause ─┤
                             └─→ replanner ◄───────┘
                                    │
                        continue/replan → executor
                        finish        → END
```

* **planner** — на основі запиту користувача формує структурований `Plan`
  (`goal` + список кроків) через `with_structured_output`.
* **executor** — виконує рівно один крок поточного плану: обирає tool
  (`bind_tools`), викликає його або, якщо tool ризиковий, зупиняється перед
  виконанням.
* **approval** — `interrupt()`-вузол, що очікує рішення людини:
  `approve` / `reject` / `edit` — для tools зі списку `RISKY_TOOLS`.
* **checkpoint_pause** — навмисний `interrupt()` після першого кроку, що
  демонструє відновлення стану з SQLite в новому процесі.
* **replanner** — після кожного кроку вирішує: продовжити виконання поточного
  плану (`continue`), змінити залишкові кроки (`replan`) чи завершити
  (`finish`).

Стан графа (`PlanExecuteState`) зберігається через `SqliteSaver` у файл
`agent_state.db`, тому кожен `thread_id` має власну, незалежну історію.

## 3. Структура файлів

```
tools.py                 # Звичайні tools з Pydantic-схемами та валідацією
                          # (calculate_trip_budget, estimate_hotel_cost, recommend_transport)
knowledge.py              # ChromaDB + tool search_knowledge (Agentic RAG)
hitl.py                   # Ризиковий tool book_hotel + Pydantic-схема
tool_utils.py              # JSON-контракт tools: success_json/error_json/safe_tool_invoke
react_agent.py             # LangGraph ReAct-агент: agent + tools + guardrails + JSON-лог + CLI
plan_execute.py            # LangGraph Plan-and-Execute: planner + executor + replanner + HITL + CLI
compare_agents.py          # Числове порівняння ReAct vs Plan-and-Execute на одній задачі
agent_state.db            # SQLite зі збереженим станом (генерується автоматично)
chroma_db/                # Локальна векторна база ChromaDB (генерується автоматично)
logs/                     # JSON-траєкторії, лог порівнянь (генерується автоматично)
graphs/                   # Mermaid-діаграми графів (генерується автоматично)
tests/                    # pytest-тести (валідація схем, tools, ReAct-цикл)
requirements.txt          # Залежності Python
README.md                 # Цей файл
```

## 4. Встановлення

```bash
pip install -r requirements.txt
```

Створіть файл `.env` у корені проєкту:

```
GOOGLE_API_KEY=ваш_ключ_google_generative_ai
```

Модель за замовчуванням — `gemini-3.5-flash-lite` (задається в `react_agent.py` та `plan_execute.py`).

## 5. Інструкція запуску

### Завдання 0 — ReAct-агент

```bash
python react_agent.py run "Порахуй бюджет подорожі для двох людей на 5 днів, 80 євро на день."
python react_agent.py demo    # 3 демонстраційні запити

python react_agent.py arun "..."   # той самий запит асинхронно (app.ainvoke())
python react_agent.py ademo        # послідовно vs паралельно (asyncio.gather) — з таймінгом

python react_agent.py graph        # Mermaid-діаграма графа -> graphs/react_agent.mmd
```

Кожен запуск виводить траєкторію в консоль і зберігає її у
`logs/react_<thread_id>_<timestamp>.json`.

Усі інші сценарії запускаються через `plan_execute.py`. Список команд також
доступний за допомогою `python plan_execute.py` (без аргументів). Для
Plan-and-Execute графа доступна аналогічна команда `python plan_execute.py graph`
(`graphs/plan_execute.mmd`).

### Завдання 1 — Plan-and-Execute

```bash
python plan_execute.py simple    # один крок плану
python plan_execute.py medium    # кілька tools
python plan_execute.py complex   # повний сценарій подорожі
python plan_execute.py demo      # усі три приклади підряд
```

### Завдання 2 — Checkpointer (persistence)

```bash
python plan_execute.py start     # запускає workflow і зупиняє його
                                  # після першого виконаного кроку
python plan_execute.py resume    # у НОВОМУ Python-процесі відновлює
                                  # той самий thread_id з agent_state.db
python plan_execute.py threads   # показує, що різні thread_id мають
                                  # незалежний стан
```

### Завдання 3 — Agentic RAG

```bash
python plan_execute.py rag
```

Запускає три приклади: запит, де `search_knowledge` не потрібен; запит, де
він потрібен; і запит, що поєднує звичайний tool із пошуком у базі знань.
Вибір tool повністю залишається за LLM.

### Завдання 4 — Human-in-the-Loop

```bash
python plan_execute.py hitl hitl-approve-001
python plan_execute.py approve hitl-approve-001

python plan_execute.py hitl hitl-reject-001
python plan_execute.py reject hitl-reject-001

python plan_execute.py hitl hitl-edit-001
python plan_execute.py edit hitl-edit-001
```

Кожен сценарій запускається з власним `thread_id`: перша команда доводить
graph до зупинки `interrupt_before="approval"` перед `book_hotel`, друга —
записує рішення людини через `app.update_state()` і відновлює виконання
`app.invoke(None, config)` (підтвердити, відхилити або змінити параметри
бронювання). Деталі механізму — розділ 8.

## 6. Tools

| Tool | Призначення | Ризиковий | ReAct | Plan-and-Execute |
|---|---|---|---|---|
| `calculate_trip_budget` | Розрахунок загального бюджету подорожі | ні | ✓ | ✓ |
| `estimate_hotel_cost` | Розрахунок вартості проживання | ні | ✓ | ✓ |
| `recommend_transport` | Рекомендація транспорту за відстанню та пріоритетом | ні | ✓ | ✓ |
| `search_knowledge` | Agentic RAG-пошук у ChromaDB (страхування, документи, багаж, правила) | ні | ✓ | ✓ |
| `book_hotel` | Фактичне бронювання готелю | **так — потребує HITL approval** | — | ✓ |

Усі tools мають Pydantic `args_schema` з валідаторами (діапазони значень,
формат дати, довжина рядків тощо). `travelers` (кількість мандрівників)
валідується спільним `Annotated`-типом `TravelersCount`, який використовують
одразу `TripBudgetInput` і `TransportInput`, щоб не дублювати логіку.

**JSON-контракт результату.** Кожен tool повертає JSON-рядок стандартної
структури (хелпери `tool_utils.success_json()` / `tool_utils.error_json()`):

```json
{"status": "success", "data": {...}}
{"status": "error", "error": "..."}
```

Помилки *всередині* тіла tool-функції (business-логіка) перехоплюються й
повертаються як `{"status": "error", ...}`. Помилки Pydantic-валідації
`args_schema`, які LangChain піднімає ще ДО виклику функції (наприклад,
коли LLM передав некоректні аргументи), перехоплюються на рівні виклику
через `tool_utils.safe_tool_invoke()` — і `executor_node`
(`plan_execute.py`), і `tools_node` (`react_agent.py`) викликають tools
саме через цей хелпер, тому жоден невдалий виклик tool не "валить" граф
цілком: помилка повертається як звичайний JSON-результат кроку, і LLM
(або replanner) бачить її та може відреагувати.

## 7. База знань (ChromaDB)

`knowledge.py` створює локальну, персистентну колекцію ChromaDB
(`./chroma_db`, колекція `travel_knowledge`) і заповнює її 10 короткими
документами доменної області "подорожі". Кожен документ має `id`, `title` та
текст; `initialize_knowledge_base()` додає лише ті документи, яких ще немає
в колекції, тому повторний імпорт `knowledge.py` не створює дублікатів.

| id | title |
|---|---|
| travel-001 | Travel insurance |
| travel-002 | Airport arrival |
| travel-003 | Cabin baggage |
| travel-004 | Hotel check-in |
| travel-005 | Emergency budget |
| travel-006 | Train travel |
| travel-007 | Flight travel |
| travel-008 | Travel documents |
| travel-009 | Hotel cancellation |
| travel-010 | Local public transport |

`search_knowledge` (Pydantic-схема `KnowledgeSearchInput`: `query` мінімум
3 символи, `top_k` від 1 до 5) виконує `collection.query()` за
семантичною близькістю і повертає `top_k` найрелевантніших фрагментів у
форматі `"{title}: {text}"`. Executor викликає цей tool лише тоді, коли
LLM сам вирішує, що для поточного кроку плану потрібна довідкова
інформація, а не розрахунок (детальніше — розділ 9).

## 8. Ризиковий tool та HITL flow (`interrupt_before`)

`book_hotel` (`hitl.py`) — єдиний tool у списку `RISKY_TOOLS`
(`plan_execute.py`). Коли executor обирає ризиковий tool, він **не викликає
його одразу**, а зберігає `pending_tool_call` у стані графа і передає
керування вузлу `approval`.

Граф скомпільований з `interrupt_before=["approval"]`:

```python
app = graph.compile(
    checkpointer=saver,
    interrupt_before=["approval"],
)
```

Це означає, що виконання зупиняється **ще ДО того**, як `approval_node`
взагалі почне виконуватись — жодного явного виклику `interrupt()`
усередині node не потрібно. `app.invoke(...)` повертається одразу, як
тільки граф підходить до `"approval"`; деталі ризикової дії читаються
через `app.get_state(config).values["pending_tool_call"]`
(`print_pending_approval()` у `plan_execute.py`).

Продовження виконання відбувається у два кроки з тим самим `thread_id`:

```python
app.update_state(config, {"human_decision": {"action": "approve"}})
result = app.invoke(None, config=config)   # None = "продовжити з місця зупинки"
```

Рішення людини записується в state ЗОВНІ, до відновлення графа; коли
`approval_node` нарешті виконується, він читає `state["human_decision"]`
і діє відповідно:

* **approve** — `safe_tool_invoke(book_hotel, original_args)` виконується
  без змін; JSON-результат (з `booking_id`) додається до `results`,
  `replanner` зазвичай завершує задачу (`finish`).
* **reject** — tool **не викликається**; замість цього в `results`
  записується повідомлення про відмову (за наявності — з причиною
  `reason`). `replanner` бачить, що ризикову дію скасовано, і завершує
  виконання (`finish`), не намагаючись повторити `book_hotel`.
* **edit** — людина передає нові `args` у `human_decision`;
  `safe_tool_invoke(book_hotel, edited_args)` виконується з оновленими
  параметрами (Pydantic-валідація `HotelBookingInput` спрацьовує повторно
  всередині `tool.invoke()`, а `safe_tool_invoke` перетворює можливу
  помилку валідації на JSON-помилку замість краху графа).

Граф можна навіть візуалізувати — `python plan_execute.py graph`
автоматично позначає вузол `approval` міткою `__interrupt = before` на
Mermaid-діаграмі (`graphs/plan_execute.mmd`).

## 9. Аналіз результатів

Нижче — спостереження з реальних запусків кожного сценарію (без
редагування чи вигаданих цифр).

**Plan-and-Execute.** У simple-прикладі planner коректно згенерував план з
одного кроку і executor одразу обрав `calculate_trip_budget` з правильними
аргументами (`{"travelers": 2, "days": 5, "daily_budget": 80}` → `€800.00`).
У medium/complex-прикладах planner будує 2-3 кроки, а executor послідовно
викликає `recommend_transport` → `estimate_hotel_cost` → `calculate_trip_budget`,
не забігаючи наперед (одна дія за одну ітерацію).

**Checkpointer.** Команда `start` зупинила workflow одразу після кроку 1
(`recommend_transport`) через `checkpoint_pause`. У **новому** Python-процесі
команда `resume` прочитала стан із `agent_state.db` (`current_step=1`, план і
результати кроку 1 — незмінні) і продовжила виконання: `replanner` обрав
`continue`, executor виконав кроки 2 і 3, а фінальний `results` містив усі
три кроки в правильному порядку. Команда `threads` підтвердила ізоляцію:
`checkpoint-session-001` містив повний стан, `checkpoint-session-002` (інший
`thread_id`) — порожній `{}`.

**Agentic RAG.** На запиті "порахуй бюджет" (без згадки довідкової
інформації) агент викликав лише `calculate_trip_budget` і жодного разу не
торкнувся `search_knowledge`. На запиті "що перевірити перед міжнародною
подорожжю" — навпаки, викликав лише `search_knowledge` і повернув релевантні
документи (`Travel documents`, `Travel insurance`, `Cabin baggage`). На
комбінованому запиті ("порахуй бюджет і скажи, що перевірити") planner
самостійно склав план із двох кроків і executor викликав обидва tools у
правильному порядку. Вибір жодного разу не потребував додаткових підказок
у коді — лише опис tools у промпті planner/executor.

**HITL.** У approve-сценарії `book_hotel` виконався тільки після
`Command(resume={"action": "approve"})` і повернув `Booking ID:
DEMO-BOOKING-001`. У reject-сценарії `book_hotel` **не викликався** —
`results` містив рядок "Ризикову дію відхилено користувачем. Причина:
...", а `replanner` після цього одразу прийняв рішення `finish`, коректно
розпізнавши, що повторювати відхилену дію не потрібно.

**ReAct-агент.** На запиті "порахуй бюджет подорожі" агент викликав
`calculate_trip_budget` рівно один раз і одразу після отримання результату
дав фінальну відповідь (2 звернення до LLM, без tool_calls на другому кроці) —
JSON-траєкторія цього запуску збережена у `logs/`. Guardrails перевірені
тестами `tests/test_react_agent.py` через фейкову LLM: `max_steps=10`
гарантовано зупиняє граф, коли LLM намагається викликати tool у нескінченному
циклі без повторів аргументів, а detекція повторів блокує повторний виклик
того самого tool з тими самими аргументами і форсовано завершує граф після
трьох таких спроб.

## 10. Тести (pytest)

```bash
python -m pytest tests/ -v
```

Тести не потребують мережі чи `GOOGLE_API_KEY` — ReAct-цикл перевіряється
через фейкову chat-модель (`FakeToolCallingModel` у `tests/test_react_agent.py`),
яка підміняється замість Gemini через ін'єкцію `build_app(chat_model=...)`.

| Файл | Що перевіряє |
|---|---|
| `tests/test_tools_validation.py` | Pydantic-схеми всіх 5 tools: коректні та некоректні вхідні дані (≥5 tests на схему) |
| `tests/test_tools_execution.py` | Виконання tools через `.invoke()`: коректний результат та помилки валідації |
| `tests/test_react_agent.py` | Чисті guardrail-функції (`is_repeated_call`, `make_call_signature`) + базовий ReAct-цикл LLM → tool → LLM, `max_steps`, detекція повторів |

## 11. Async-виконання та візуалізація графа

**Async.** `react_agent.py` підтримує `app.ainvoke()` без змін до вузлів
графа (LangGraph сам виконує синхронні node-функції в executor-пулі):

```bash
python react_agent.py arun "<запит>"   # одиночний асинхронний запуск
python react_agent.py ademo            # порівняння часу: цикл invoke() vs asyncio.gather(ainvoke())
```

`ademo` запускає 3 демо-запити спершу послідовно (`app.invoke()` у циклі),
потім паралельно (`asyncio.gather(*[app.ainvoke(...) for ...])`) і друкує
пряме порівняння часу виконання — практична демонстрація, а не лише факт
підтримки `async`.

**Візуалізація.** Обидва графи можна експортувати як Mermaid-діаграму
(`get_graph().draw_mermaid()`, без зовнішніх залежностей):

```bash
python react_agent.py graph      # -> graphs/react_agent.mmd
python plan_execute.py graph     # -> graphs/plan_execute.mmd
```

Діаграма Plan-and-Execute автоматично позначає вузол `approval` міткою
`__interrupt = before`, що візуально підтверджує механізм HITL із
розділу 8.

## 12. Числове порівняння: ReAct vs Plan-and-Execute

```bash
python compare_agents.py
```

Скрипт `compare_agents.py` запускає **той самий** запит (розрахунок
бюджету + вартості готелю + пошук довідкової інформації, без `book_hotel`
— цей tool є лише у Plan-and-Execute, тому виключений із чесного
порівняння) через обидва агенти й вимірює: час виконання, кількість
LLM-викликів, кількість tool-викликів. Результат зберігається у
`logs/agent_comparison.json`.

Результат реального запуску (без редагування чи вигаданих цифр —
абсолютний час залежить від навантаження Google Gemini API на момент
запуску, див. розділ 13):

| Агент | Час (с) | LLM-викликів | Tool-викликів |
|---|---|---|---|
| ReAct | 236.34 | 4 | 3 |
| Plan-and-Execute | 476.18 | 7 | 3 |

Обидва агенти виконали задачу повністю й дали коректну фінальну
відповідь із трьома tool-викликами (`estimate_hotel_cost`,
`calculate_trip_budget`, `search_knowledge`). ReAct витратив менше
LLM-викликів (4 проти 7), бо кожна ітерація одразу вирішує "tool чи
відповідь", тоді як Plan-and-Execute додає окремий LLM-виклик на
`replanner` після кожного кроку (тут — 3 кроки → 3 replanner-виклики +
1 planner-виклик + 3 executor-виклики = 7). Це ілюструє типовий
trade-off: ReAct дешевший за кількістю LLM-викликів на простих
лінійних задачах, а Plan-and-Execute платить додаткові виклики за
явний контроль прогресу (`continue`/`replan`/`finish`) на кожному
кроці.

## 13. Відомі обмеження

* Tools використовують локальні правила та розрахунки замість реальних travel API.
* `recommend_transport` базується на спрощених правилах відстані/пріоритету.
* Робота агента залежить від доступності та квот Google Gemini API.
* `search_knowledge` працює на невеликому, вручну підготовленому наборі документів.
* ReAct-агент не підключений до `book_hotel` — ризикова дія та HITL
  демонструються лише в Plan-and-Execute (`plan_execute.py`).
* Абсолютні числа в розділі 12 залежать від поточного навантаження та
  rate limits Google Gemini API на момент запуску; для відтворюваності
  запускайте `compare_agents.py` самостійно, а не покладайтесь лише на
  записані значення в README.
