from typing import Annotated

from langchain_core.tools import tool
from pydantic import AfterValidator, BaseModel, Field, field_validator

from tool_utils import error_json, success_json


# ================================================================
# Спільні валідатори
# ================================================================

def _validate_travelers_range(value: int) -> int:
    if not 1 <= value <= 10:
        raise ValueError("Кількість мандрівників повинна бути від 1 до 10.")
    return value


TravelersCount = Annotated[int, AfterValidator(_validate_travelers_range)]


# ================================================================
# TOOL 1 — Розрахунок бюджету подорожі
# ================================================================

class TripBudgetInput(BaseModel):
    """Параметри для розрахунку бюджету подорожі."""

    travelers: TravelersCount = Field(
        description="Кількість мандрівників. Допустиме значення: від 1 до 10."
    )
    days: int = Field(
        description="Тривалість подорожі у днях. Допустиме значення: від 1 до 30."
    )
    daily_budget: float = Field(
        description="Щоденний бюджет на одну людину в євро."
    )

    @field_validator("days")
    @classmethod
    def validate_days(cls, value: int) -> int:
        if not 1 <= value <= 30:
            raise ValueError("Кількість днів повинна бути від 1 до 30.")
        return value

    @field_validator("daily_budget")
    @classmethod
    def validate_daily_budget(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("Щоденний бюджет повинен бути більшим за 0.")
        return value


@tool(args_schema=TripBudgetInput)
def calculate_trip_budget(
    travelers: int,
    days: int,
    daily_budget: float,
) -> str:
    """Розрахувати орієнтовний бюджет подорожі.

    Використовуй цей інструмент, коли користувач хоче дізнатися,
    скільки приблизно коштуватиме поїздка з урахуванням кількості
    мандрівників, тривалості подорожі та щоденного бюджету.

    Args:
        travelers: Кількість мандрівників.
        days: Кількість днів подорожі.
        daily_budget: Щоденний бюджет на одну людину в євро.

    Returns:
        JSON-рядок {"status": "success", "data": {...}} з розрахованим
        бюджетом або {"status": "error", "error": "..."} у разі помилки.
    """

    try:
        total = travelers * days * daily_budget

        return success_json(
            {
                "total_budget": round(total, 2),
                "currency": "EUR",
                "travelers": travelers,
                "days": days,
                "daily_budget": daily_budget,
            }
        )
    except Exception as exc:
        return error_json(f"Не вдалося розрахувати бюджет: {exc}")


# ================================================================
# TOOL 2 — Розрахунок вартості готелю
# ================================================================

class HotelCostInput(BaseModel):
    """Параметри для розрахунку вартості проживання."""

    nights: int = Field(
        description="Кількість ночей у готелі. Допустиме значення: від 1 до 30."
    )
    price_per_night: float = Field(
        description="Вартість одного номера за одну ніч у євро."
    )
    rooms: int = Field(
        default=1,
        description="Кількість номерів. Допустиме значення: від 1 до 5."
    )

    @field_validator("nights")
    @classmethod
    def validate_nights(cls, value: int) -> int:
        if not 1 <= value <= 30:
            raise ValueError("Кількість ночей повинна бути від 1 до 30.")
        return value

    @field_validator("price_per_night")
    @classmethod
    def validate_price(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("Вартість ночі повинна бути більшою за 0.")
        return value

    @field_validator("rooms")
    @classmethod
    def validate_rooms(cls, value: int) -> int:
        if not 1 <= value <= 5:
            raise ValueError("Кількість номерів повинна бути від 1 до 5.")
        return value


@tool(args_schema=HotelCostInput)
def estimate_hotel_cost(
    nights: int,
    price_per_night: float,
    rooms: int = 1,
) -> str:
    """Розрахувати загальну вартість проживання в готелі.

    Використовуй цей інструмент, коли користувач хоче оцінити
    витрати на готель за відомою кількістю ночей, номерів
    та ціною одного номера за ніч.

    Args:
        nights: Кількість ночей.
        price_per_night: Вартість одного номера за ніч.
        rooms: Кількість номерів.

    Returns:
        JSON-рядок {"status": "success", "data": {...}} з вартістю
        проживання або {"status": "error", "error": "..."} у разі помилки.
    """

    try:
        total = nights * price_per_night * rooms

        return success_json(
            {
                "total_cost": round(total, 2),
                "currency": "EUR",
                "nights": nights,
                "price_per_night": price_per_night,
                "rooms": rooms,
            }
        )
    except Exception as exc:
        return error_json(f"Не вдалося розрахувати вартість готелю: {exc}")


# ================================================================
# TOOL 3 — Рекомендація транспорту
# ================================================================

class TransportInput(BaseModel):
    """Параметри для вибору рекомендованого транспорту."""

    distance_km: float = Field(
        description="Відстань між пунктами подорожі у кілометрах."
    )
    travelers: TravelersCount = Field(
        default=1,
        description="Кількість мандрівників. Допустиме значення: від 1 до 10."
    )
    priority: str = Field(
        default="balanced",
        description="Пріоритет подорожі: cheap, fast або balanced."
    )

    @field_validator("distance_km")
    @classmethod
    def validate_distance(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("Відстань повинна бути більшою за 0.")
        if value > 20000:
            raise ValueError("Відстань не може перевищувати 20000 км.")
        return value

    @field_validator("priority")
    @classmethod
    def validate_priority(cls, value: str) -> str:
        value = value.strip().lower()

        allowed = {"cheap", "fast", "balanced"}

        if value not in allowed:
            raise ValueError(
                "Пріоритет повинен бути 'cheap', 'fast' або 'balanced'."
            )

        return value


@tool(args_schema=TransportInput)
def recommend_transport(
    distance_km: float,
    travelers: int = 1,
    priority: str = "balanced",
) -> str:
    """Порекомендувати вид транспорту для подорожі.

    Використовуй цей інструмент, коли користувач хоче отримати
    рекомендацію щодо транспорту залежно від відстані та пріоритету:
    дешевше, швидше або збалансований варіант.

    Args:
        distance_km: Відстань у кілометрах.
        travelers: Кількість мандрівників.
        priority: Пріоритет cheap, fast або balanced.

    Returns:
        JSON-рядок {"status": "success", "data": {...}} з рекомендованим
        транспортом або {"status": "error", "error": "..."} у разі помилки.
    """

    try:
        if distance_km < 10:
            transport = "громадський транспорт або таксі"

        elif distance_km < 300:
            if priority == "cheap":
                transport = "автобус"
            elif priority == "fast":
                transport = "потяг"
            else:
                transport = "потяг або автобус"

        elif distance_km < 1000:
            if priority == "cheap":
                transport = "автобус або потяг"
            elif priority == "fast":
                transport = "літак"
            else:
                transport = "потяг"

        else:
            transport = "літак"

        return success_json(
            {
                "recommended_transport": transport,
                "distance_km": distance_km,
                "travelers": travelers,
                "priority": priority,
            }
        )
    except Exception as exc:
        return error_json(f"Не вдалося порекомендувати транспорт: {exc}")

# ================================================================
# UNIT TESTS
# ================================================================

if __name__ == "__main__":

    # ── Тест Tool 1: calculate_trip_budget ──────────────────────

    # Перевіряємо Pydantic-валідацію
    try:
        TripBudgetInput(
            travelers=0,
            days=5,
            daily_budget=80
        )  # Повинно впасти
    except Exception as e:
        print(f"Validation error (expected): {e}")

    # Перевіряємо роботу tool
    result = calculate_trip_budget.invoke(
        {
            "travelers": 2,
            "days": 5,
            "daily_budget": 80
        }
    )
    print(f"Trip budget: {result}")


    # ── Тест Tool 2: estimate_hotel_cost ────────────────────────

    # Перевіряємо Pydantic-валідацію
    try:
        HotelCostInput(
            nights=0,
            price_per_night=100,
            rooms=1
        )  # Повинно впасти
    except Exception as e:
        print(f"Validation error (expected): {e}")

    # Перевіряємо роботу tool
    result = estimate_hotel_cost.invoke(
        {
            "nights": 5,
            "price_per_night": 100,
            "rooms": 1
        }
    )
    print(f"Hotel cost: {result}")


    # ── Тест Tool 3: recommend_transport ────────────────────────

    # Перевіряємо Pydantic-валідацію
    try:
        TransportInput(
            distance_km=-100,
            travelers=2,
            priority="fast"
        )  # Повинно впасти
    except Exception as e:
        print(f"Validation error (expected): {e}")

    # Перевіряємо роботу tool
    result = recommend_transport.invoke(
        {
            "distance_km": 500,
            "travelers": 2,
            "priority": "fast"
        }
    )
    print(f"Transport: {result}")