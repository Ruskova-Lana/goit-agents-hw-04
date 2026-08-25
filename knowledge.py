import chromadb

from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator

from tool_utils import error_json, success_json


# ================================================================
# Документи бази знань
# ================================================================

KNOWLEDGE_DOCUMENTS = [
    {
        "id": "travel-001",
        "title": "Travel insurance",
        "text": (
            "Для міжнародних подорожей рекомендується мати "
            "туристичне страхування, яке покриває невідкладну "
            "медичну допомогу, госпіталізацію та репатріацію."
        ),
    },
    {
        "id": "travel-002",
        "title": "Airport arrival",
        "text": (
            "Для міжнародних рейсів зазвичай рекомендується "
            "прибути до аеропорту приблизно за 2-3 години "
            "до запланованого вильоту."
        ),
    },
    {
        "id": "travel-003",
        "title": "Cabin baggage",
        "text": (
            "Правила ручної поклажі залежать від авіакомпанії. "
            "Перед вильотом необхідно перевірити допустимі "
            "розміри, вагу та обмеження перевізника."
        ),
    },
    {
        "id": "travel-004",
        "title": "Hotel check-in",
        "text": (
            "Типовий час check-in у готелях — після 14:00 або 15:00. "
            "Check-out зазвичай відбувається до 11:00 або 12:00."
        ),
    },
    {
        "id": "travel-005",
        "title": "Emergency budget",
        "text": (
            "При плануванні подорожі корисно передбачити резерв "
            "приблизно 10-15 відсотків бюджету на непередбачені витрати."
        ),
    },
    {
        "id": "travel-006",
        "title": "Train travel",
        "text": (
            "Потяг часто є зручним транспортом для середніх відстаней, "
            "особливо коли залізничні вокзали розташовані "
            "безпосередньо в центрі міст."
        ),
    },
    {
        "id": "travel-007",
        "title": "Flight travel",
        "text": (
            "Літак зазвичай є найшвидшим варіантом для великих відстаней, "
            "але потрібно враховувати трансфер до аеропорту, "
            "реєстрацію та проходження контролю безпеки."
        ),
    },
    {
        "id": "travel-008",
        "title": "Travel documents",
        "text": (
            "Перед міжнародною подорожжю необхідно перевірити "
            "термін дії паспорта, правила в'їзду, "
            "візові вимоги та інші необхідні документи."
        ),
    },
    {
        "id": "travel-009",
        "title": "Hotel cancellation",
        "text": (
            "Перед бронюванням готелю слід перевірити умови скасування. "
            "Non-refundable тариф часто дешевший, "
            "але зазвичай не дозволяє повернути кошти."
        ),
    },
    {
        "id": "travel-010",
        "title": "Local public transport",
        "text": (
            "У великих містах громадський транспорт зазвичай дешевший "
            "за таксі. Денний або туристичний проїзний може бути "
            "вигідним при великій кількості поїздок."
        ),
    },
]


# ================================================================
# ChromaDB
# ================================================================

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_or_create_collection(
    name="travel_knowledge"
)


# ================================================================
# Ініціалізація бази знань
# ================================================================

def initialize_knowledge_base() -> None:
    """Додає документи до ChromaDB, якщо їх ще немає."""

    existing = collection.get()

    existing_ids = set(
        existing.get("ids", [])
    )

    documents_to_add = [
        document
        for document in KNOWLEDGE_DOCUMENTS
        if document["id"] not in existing_ids
    ]

    if not documents_to_add:
        return


    collection.add(
        ids=[
            document["id"]
            for document in documents_to_add
        ],

        documents=[
            document["text"]
            for document in documents_to_add
        ],

        metadatas=[
            {
                "title": document["title"]
            }
            for document in documents_to_add
        ],
    )


# При імпорті knowledge.py база ініціалізується автоматично
initialize_knowledge_base()


# ================================================================
# Pydantic schema
# ================================================================

class KnowledgeSearchInput(BaseModel):
    """Параметри пошуку у базі знань."""

    query: str = Field(
        description=(
            "Запит для пошуку інформації у внутрішній "
            "базі знань про подорожі."
        )
    )

    top_k: int = Field(
        default=3,
        description=(
            "Кількість найбільш релевантних документів "
            "від 1 до 5."
        )
    )


    @field_validator("query")
    @classmethod
    def validate_query(
        cls,
        value: str,
    ) -> str:

        value = value.strip()

        if len(value) < 3:

            raise ValueError(
                "Пошуковий запит повинен містити "
                "щонайменше 3 символи."
            )

        return value


    @field_validator("top_k")
    @classmethod
    def validate_top_k(
        cls,
        value: int,
    ) -> int:

        if not 1 <= value <= 5:

            raise ValueError(
                "top_k повинен бути від 1 до 5."
            )

        return value


# ================================================================
# Agentic RAG tool
# ================================================================

@tool(args_schema=KnowledgeSearchInput)
def search_knowledge(
    query: str,
    top_k: int = 3,
) -> str:
    """Знайти інформацію у внутрішній travel knowledge base.

    Використовуй цей tool, коли для виконання поточного кроку
    потрібні довідкові знання про:

    - документи;
    - візові вимоги;
    - страхування;
    - багаж;
    - правила подорожей;
    - готельні правила;
    - рекомендації щодо транспорту;
    - підготовку до міжнародної поїздки.

    Не використовуй цей інструмент для простих математичних
    розрахунків або задач, для яких існує спеціалізований tool.

    Args:
        query: Запит до бази знань.
        top_k: Кількість документів для повернення.

    Returns:
        JSON-рядок {"status": "success", "data": {"results": [...]}} з
        найбільш релевантними фрагментами бази знань або
        {"status": "error", "error": "..."} у разі помилки.
    """

    try:
        results = collection.query(
            query_texts=[query],
            n_results=top_k,
        )

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