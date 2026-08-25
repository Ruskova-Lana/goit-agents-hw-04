from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator

from tool_utils import error_json, success_json


# ================================================================
# Pydantic schema для ризикового tool
# ================================================================

class HotelBookingInput(BaseModel):
    """Параметри бронювання готелю."""

    hotel_name: str = Field(
        description="Назва готелю."
    )

    check_in: str = Field(
        description="Дата заїзду у форматі YYYY-MM-DD."
    )

    nights: int = Field(
        description="Кількість ночей."
    )

    total_cost: float = Field(
        description="Загальна вартість бронювання у EUR."
    )


    @field_validator("hotel_name")
    @classmethod
    def validate_hotel_name(
        cls,
        value: str,
    ) -> str:

        value = value.strip()

        if len(value) < 2:
            raise ValueError(
                "Назва готелю повинна містити щонайменше 2 символи."
            )

        return value


    @field_validator("check_in")
    @classmethod
    def validate_check_in(
        cls,
        value: str,
    ) -> str:

        value = value.strip()

        parts = value.split("-")

        if (
            len(parts) != 3
            or len(parts[0]) != 4
            or len(parts[1]) != 2
            or len(parts[2]) != 2
        ):
            raise ValueError(
                "Дата повинна бути у форматі YYYY-MM-DD."
            )

        return value


    @field_validator("nights")
    @classmethod
    def validate_nights(
        cls,
        value: int,
    ) -> int:

        if not 1 <= value <= 30:
            raise ValueError(
                "Кількість ночей повинна бути від 1 до 30."
            )

        return value


    @field_validator("total_cost")
    @classmethod
    def validate_total_cost(
        cls,
        value: float,
    ) -> float:

        if value <= 0:
            raise ValueError(
                "Вартість бронювання повинна бути більшою за 0."
            )

        return value


# ================================================================
# Ризиковий tool
# ================================================================

@tool(args_schema=HotelBookingInput)
def book_hotel(
    hotel_name: str,
    check_in: str,
    nights: int,
    total_cost: float,
) -> str:
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