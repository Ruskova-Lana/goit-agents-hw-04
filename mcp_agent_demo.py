"""LangGraph-агент, підключений до mcp_server.py через MultiServerMCPClient.

Демонструє інтеграцію MCP (Завд. 3) з LangGraph: замість LangChain `@tool`
об'єктів з tools.py, агент отримує tools через стандартизований MCP-протокол
(stdio-транспорт — `mcp_server.py` запускається як subprocess).

python mcp_agent_demo.py demo      # 3+ запити через MCP-агента
python mcp_agent_demo.py resources # прочитати обидва MCP resources
python mcp_agent_demo.py prompts   # заповнити обидва MCP prompts
"""

import asyncio
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent


load_dotenv()


SYSTEM_PROMPT = (
    "Ти туристичний AI-агент. Усі твої інструменти (розрахунок бюджету, "
    "вартості готелю, рекомендація транспорту, конвертація валют, пошук у "
    "базі знань) підключені через MCP-сервер. Викликай tool лише тоді, "
    "коли він дійсно потрібен, і давай коротку фінальну відповідь на "
    "основі отриманих результатів."
)


def build_mcp_client() -> MultiServerMCPClient:
    """MCP-клієнт, що піднімає mcp_server.py як stdio subprocess."""

    return MultiServerMCPClient(
        {
            "travel": {
                "transport": "stdio",
                "command": sys.executable,
                "args": ["mcp_server.py"],
            }
        }
    )


DEMO_QUERIES = [
    "Я їду удвох на 5 днів. Щоденний бюджет на одну людину — 80 євро. "
    "Порахуй загальний бюджет подорожі.",
    "Порекомендуй транспорт для подорожі на 800 км, головний пріоритет — швидкість.",
    "Що потрібно перевірити перед міжнародною подорожжю? Скористайся базою знань.",
    "Скільки буде 250 євро в доларах США?",
]


async def run_demo() -> None:
    client = build_mcp_client()
    tools = await client.get_tools()

    print("=" * 70)
    print(f"MCP tools, підключені через MultiServerMCPClient: {[t.name for t in tools]}")
    print("=" * 70)

    model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0.1)
    agent = create_agent(model, tools, system_prompt=SYSTEM_PROMPT)

    for query in DEMO_QUERIES:
        print("\n" + "#" * 80)
        print(f"USER: {query}")
        print("#" * 80)

        result = await agent.ainvoke({"messages": [HumanMessage(content=query)]})

        for message in result["messages"]:
            tool_calls = getattr(message, "tool_calls", None)
            if tool_calls:
                for call in tool_calls:
                    print(f"  [tool_call] {call['name']}({call['args']})")
            if getattr(message, "name", None):
                print(f"  [tool_result:{message.name}] {str(message.content)[:200]}")

        final_message = result["messages"][-1]
        print(f"\nFINAL ANSWER:\n{final_message.content}")


async def run_resources_demo() -> None:
    client = build_mcp_client()
    resources = await client.get_resources("travel")

    print("=" * 70)
    print("MCP RESOURCES (travel://knowledge-base, travel://currency-rates)")
    print("=" * 70)
    for blob in resources:
        print(f"\n--- {blob.metadata.get('uri', blob.source)} ---")
        print(blob.as_string()[:500])


async def run_prompts_demo() -> None:
    client = build_mcp_client()

    trip_messages = await client.get_prompt(
        "travel",
        "trip_planning",
        arguments={"destination": "Lisbon", "days": "5", "travelers": "2", "priority": "fast"},
    )
    budget_messages = await client.get_prompt(
        "travel",
        "budget_summary",
        arguments={"trip_budget": "800", "hotel_cost": "400", "currency": "EUR"},
    )

    print("=" * 70)
    print("MCP PROMPT — trip_planning")
    print("=" * 70)
    for m in trip_messages:
        print(m.content)

    print("\n" + "=" * 70)
    print("MCP PROMPT — budget_summary")
    print("=" * 70)
    for m in budget_messages:
        print(m.content)


def print_help() -> None:
    print(
        """
============================================================
MCP + LangGraph integration demo (ДЗ4 Завд. 3)
============================================================

python mcp_agent_demo.py demo        4 запити через create_react_agent(),
                                      tools отримані з MultiServerMCPClient.

python mcp_agent_demo.py resources   Читає обидва MCP resources.

python mcp_agent_demo.py prompts     Заповнює обидва MCP prompts.
============================================================
"""
    )


if __name__ == "__main__":
    command = sys.argv[1].lower() if len(sys.argv) > 1 else "help"

    if command == "demo":
        asyncio.run(run_demo())
    elif command == "resources":
        asyncio.run(run_resources_demo())
    elif command == "prompts":
        asyncio.run(run_prompts_demo())
    else:
        print_help()
