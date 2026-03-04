import logging
from typing import Annotated, TypedDict, Literal

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

from core.config import settings
from tools import tools_list

logger = logging.getLogger(__name__)

# Agent singleton
_agent = None


def get_agent():
    global _agent
    if _agent is None:
        raise RuntimeError("Agent not initialized.")
    return _agent


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


SYSTEM_PROMPT = """You are QuantAI, an expert financial analyst assistant specializing in stocks and cryptocurrency.

You have access to real-time market data tools:
- get_stock_data: Fetch stock prices, technicals, OHLCV data
- get_crypto_data: Fetch crypto prices, market data from CoinGecko
- get_market_overview: Get major market indices and top crypto overview
- predict_asset: Generate technical analysis predictions
- get_news_sentiment: Fetch recent news and compute sentiment score (bullish/bearish signals)

When users ask about stocks or crypto:
1. ALWAYS use tools to get real data first
2. Check news sentiment for fundamental context
3. Combine technical + sentiment signals for superior analysis
4. Return your response as a JSON object:
{
  "message": "Your analysis in markdown format",
  "data_type": "stock" | "crypto" | "market" | "prediction" | "chat",
  "asset_data": { ... raw data from tools ... },
  "prediction": { ... prediction data if applicable ... },
  "sentiment": { ... sentiment data if analyzed ... }
}"""


# Build agent once

def _build_agent():
    """Build the LangGraph agent once at startup."""
    llm = ChatOpenAI(
        model=settings.openai_model,
        temperature=0.1,
        api_key=settings.openai_api_key,
    )
    llm_with_tools = llm.bind_tools(tools_list)

    def agent_node(state: AgentState):
        messages = state["messages"]
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
        return {"messages": [llm_with_tools.invoke(messages)]}

    def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
        last = state["messages"][-1]
        return "tools" if (hasattr(last, "tool_calls") and last.tool_calls) else "__end__"

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools_list))
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")
    compiled = graph.compile()
    logger.info("[App] LangGraph agent compiled.")
    return compiled


# called by main application on startup

def init_agent():
    global _agent
    if _agent is None:
        _agent = _build_agent()
    return _agent
