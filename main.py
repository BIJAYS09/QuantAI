"""
Stock & Crypto Prediction API
Built with FastAPI + LangGraph + LangChain
"""

import os
import json
import requests
import numpy as np
from datetime import datetime, timedelta
from typing import Annotated, TypedDict, Literal
from dotenv import load_dotenv

import yfinance as yf
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# LangChain / LangGraph imports
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

# Load env
load_dotenv()

app = FastAPI(title="Stock & Crypto Predictor API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# TOOLS
# ─────────────────────────────────────────────

@tool
def get_stock_data(symbol: str, period: str = "3mo") -> str:
    """
    Fetch historical stock price data and technical indicators for a given ticker symbol.
    Args:
        symbol: Stock ticker like AAPL, TSLA, MSFT
        period: Data period - 1mo, 3mo, 6mo, 1y
    """
    try:
        ticker = yf.Ticker(symbol.upper())
        hist = ticker.history(period=period)
        info = ticker.info

        if hist.empty:
            return json.dumps({"error": f"No data found for {symbol}"})

        # Calculate simple moving averages
        hist["SMA20"] = hist["Close"].rolling(window=20).mean()
        hist["SMA50"] = hist["Close"].rolling(window=50).mean()

        # RSI
        delta = hist["Close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss
        hist["RSI"] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = hist["Close"].ewm(span=12).mean()
        ema26 = hist["Close"].ewm(span=26).mean()
        hist["MACD"] = ema12 - ema26
        hist["Signal"] = hist["MACD"].ewm(span=9).mean()

        latest = hist.iloc[-1]
        prev = hist.iloc[-2] if len(hist) > 1 else latest
        week_ago = hist.iloc[-5] if len(hist) >= 5 else hist.iloc[0]
        month_ago = hist.iloc[-22] if len(hist) >= 22 else hist.iloc[0]

        # Build OHLCV for chart (last 60 days)
        chart_data = []
        for idx, row in hist.tail(60).iterrows():
            chart_data.append({
                "date": idx.strftime("%Y-%m-%d"),
                "open": round(float(row["Open"]), 2),
                "high": round(float(row["High"]), 2),
                "low": round(float(row["Low"]), 2),
                "close": round(float(row["Close"]), 2),
                "volume": int(row["Volume"]),
                "sma20": round(float(row["SMA20"]), 2) if not pd.isna(row["SMA20"]) else None,
                "sma50": round(float(row["SMA50"]), 2) if not pd.isna(row["SMA50"]) else None,
            })

        result = {
            "symbol": symbol.upper(),
            "name": info.get("longName", symbol.upper()),
            "current_price": round(float(latest["Close"]), 2),
            "open": round(float(latest["Open"]), 2),
            "high": round(float(latest["High"]), 2),
            "low": round(float(latest["Low"]), 2),
            "volume": int(latest["Volume"]),
            "change_1d": round(float(latest["Close"] - prev["Close"]), 2),
            "change_1d_pct": round(float((latest["Close"] - prev["Close"]) / prev["Close"] * 100), 2),
            "change_1w_pct": round(float((latest["Close"] - week_ago["Close"]) / week_ago["Close"] * 100), 2),
            "change_1m_pct": round(float((latest["Close"] - month_ago["Close"]) / month_ago["Close"] * 100), 2),
            "rsi": round(float(latest["RSI"]), 2) if not pd.isna(latest["RSI"]) else None,
            "macd": round(float(latest["MACD"]), 4) if not pd.isna(latest["MACD"]) else None,
            "macd_signal": round(float(latest["Signal"]), 4) if not pd.isna(latest["Signal"]) else None,
            "sma20": round(float(latest["SMA20"]), 2) if not pd.isna(latest["SMA20"]) else None,
            "sma50": round(float(latest["SMA50"]), 2) if not pd.isna(latest["SMA50"]) else None,
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "52w_high": info.get("fiftyTwoWeekHigh"),
            "52w_low": info.get("fiftyTwoWeekLow"),
            "sector": info.get("sector", "N/A"),
            "chart_data": chart_data,
        }
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def get_crypto_data(coin_id: str) -> str:
    """
    Fetch cryptocurrency price data from CoinGecko.
    Args:
        coin_id: CoinGecko coin ID like bitcoin, ethereum, solana, cardano, dogecoin
    """
    try:
        # Market data
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id.lower()}"
        params = {
            "localization": False,
            "tickers": False,
            "market_data": True,
            "community_data": False,
            "developer_data": False,
        }
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()

        if "error" in data:
            return json.dumps({"error": data["error"]})

        md = data.get("market_data", {})

        # Historical chart (30 days)
        chart_url = f"https://api.coingecko.com/api/v3/coins/{coin_id.lower()}/market_chart"
        chart_resp = requests.get(chart_url, params={"vs_currency": "usd", "days": 60}, timeout=10)
        chart_raw = chart_resp.json()

        chart_data = []
        prices = chart_raw.get("prices", [])
        for ts, price in prices:
            chart_data.append({
                "date": datetime.fromtimestamp(ts / 1000).strftime("%Y-%m-%d"),
                "close": round(price, 4),
            })

        result = {
            "symbol": data.get("symbol", "").upper(),
            "name": data.get("name", coin_id),
            "current_price": md.get("current_price", {}).get("usd"),
            "market_cap": md.get("market_cap", {}).get("usd"),
            "volume_24h": md.get("total_volume", {}).get("usd"),
            "change_24h": round(md.get("price_change_percentage_24h", 0), 2),
            "change_7d": round(md.get("price_change_percentage_7d", 0), 2),
            "change_30d": round(md.get("price_change_percentage_30d", 0), 2),
            "ath": md.get("ath", {}).get("usd"),
            "atl": md.get("atl", {}).get("usd"),
            "circulating_supply": md.get("circulating_supply"),
            "total_supply": md.get("total_supply"),
            "market_cap_rank": data.get("market_cap_rank"),
            "description": data.get("description", {}).get("en", "")[:500],
            "chart_data": chart_data,
        }
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def get_market_overview() -> str:
    """
    Get an overview of major market indices and top movers.
    Returns SP500, Nasdaq, Dow Jones, and top crypto prices.
    """
    try:
        # Major indices
        indices = {
            "^GSPC": "S&P 500",
            "^IXIC": "NASDAQ",
            "^DJI": "Dow Jones",
            "^VIX": "VIX",
        }
        result_indices = []
        for sym, name in indices.items():
            try:
                t = yf.Ticker(sym)
                hist = t.history(period="5d")
                if not hist.empty:
                    latest = hist.iloc[-1]["Close"]
                    prev = hist.iloc[-2]["Close"] if len(hist) > 1 else latest
                    change_pct = (latest - prev) / prev * 100
                    result_indices.append({
                        "symbol": sym,
                        "name": name,
                        "price": round(float(latest), 2),
                        "change_pct": round(float(change_pct), 2),
                    })
            except:
                pass

        # Top cryptos from CoinGecko
        crypto_url = "https://api.coingecko.com/api/v3/coins/markets"
        crypto_params = {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": 5,
            "page": 1,
            "price_change_percentage": "24h",
        }
        crypto_resp = requests.get(crypto_url, params=crypto_params, timeout=10)
        cryptos = crypto_resp.json()

        result_crypto = []
        for c in cryptos:
            result_crypto.append({
                "id": c["id"],
                "symbol": c["symbol"].upper(),
                "name": c["name"],
                "price": c["current_price"],
                "change_24h": round(c.get("price_change_percentage_24h", 0), 2),
                "market_cap": c["market_cap"],
            })

        return json.dumps({
            "indices": result_indices,
            "top_cryptos": result_crypto,
            "timestamp": datetime.now().isoformat(),
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def predict_asset(symbol: str, asset_type: str = "stock") -> str:
    """
    Generate a technical analysis-based prediction for a stock or crypto.
    Uses RSI, MACD, moving averages, and price momentum to forecast short-term movement.
    Args:
        symbol: Ticker or coin name (e.g., AAPL or bitcoin)
        asset_type: 'stock' or 'crypto'
    """
    try:
        if asset_type == "stock":
            ticker = yf.Ticker(symbol.upper())
            hist = ticker.history(period="6mo")
            if hist.empty:
                return json.dumps({"error": "No data"})
            prices = hist["Close"].values
        else:
            url = f"https://api.coingecko.com/api/v3/coins/{symbol.lower()}/market_chart"
            resp = requests.get(url, params={"vs_currency": "usd", "days": 180}, timeout=10)
            data = resp.json()
            prices = np.array([p[1] for p in data.get("prices", [])])

        if len(prices) < 50:
            return json.dumps({"error": "Not enough data for prediction"})

        # Technical signals
        prices_series = pd.Series(prices)
        sma20 = prices_series.rolling(20).mean().iloc[-1]
        sma50 = prices_series.rolling(50).mean().iloc[-1]
        current = prices[-1]

        # RSI
        delta = prices_series.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rsi = float(100 - (100 / (1 + gain.iloc[-1] / loss.iloc[-1])))

        # MACD
        ema12 = prices_series.ewm(span=12).mean()
        ema26 = prices_series.ewm(span=26).mean()
        macd = float(ema12.iloc[-1] - ema26.iloc[-1])
        signal = float((ema12 - ema26).ewm(span=9).mean().iloc[-1])

        # Bollinger Bands
        bb_mean = prices_series.rolling(20).mean().iloc[-1]
        bb_std = prices_series.rolling(20).std().iloc[-1]
        bb_upper = float(bb_mean + 2 * bb_std)
        bb_lower = float(bb_mean - 2 * bb_std)

        # Score-based prediction
        score = 0
        signals = []

        # Price vs MAs
        if current > sma20:
            score += 1
            signals.append("Price above SMA20 (bullish)")
        else:
            score -= 1
            signals.append("Price below SMA20 (bearish)")

        if sma20 > sma50:
            score += 1
            signals.append("Golden cross: SMA20 > SMA50 (bullish)")
        else:
            score -= 1
            signals.append("Death cross: SMA20 < SMA50 (bearish)")

        # RSI
        if rsi < 30:
            score += 2
            signals.append(f"RSI oversold at {rsi:.1f} — strong buy signal")
        elif rsi < 45:
            score += 1
            signals.append(f"RSI at {rsi:.1f} — mildly bullish")
        elif rsi > 70:
            score -= 2
            signals.append(f"RSI overbought at {rsi:.1f} — strong sell signal")
        elif rsi > 55:
            score -= 1
            signals.append(f"RSI at {rsi:.1f} — mildly bearish")
        else:
            signals.append(f"RSI neutral at {rsi:.1f}")

        # MACD
        if macd > signal:
            score += 1
            signals.append("MACD above signal line (bullish crossover)")
        else:
            score -= 1
            signals.append("MACD below signal line (bearish crossover)")

        # Bollinger
        if current < bb_lower:
            score += 1
            signals.append("Price below lower Bollinger Band — potential reversal up")
        elif current > bb_upper:
            score -= 1
            signals.append("Price above upper Bollinger Band — potential reversal down")

        # Momentum (5-day)
        mom_5d = float((prices[-1] - prices[-6]) / prices[-6] * 100) if len(prices) >= 6 else 0
        if mom_5d > 3:
            score += 1
            signals.append(f"Strong 5-day momentum: +{mom_5d:.1f}%")
        elif mom_5d < -3:
            score -= 1
            signals.append(f"Weak 5-day momentum: {mom_5d:.1f}%")

        # Predict
        if score >= 3:
            prediction = "STRONG BUY"
            confidence = min(90, 60 + score * 5)
            target_change = 5.0 + score * 1.5
        elif score >= 1:
            prediction = "BUY"
            confidence = min(75, 50 + score * 5)
            target_change = 2.0 + score * 1.0
        elif score <= -3:
            prediction = "STRONG SELL"
            confidence = min(90, 60 + abs(score) * 5)
            target_change = -(5.0 + abs(score) * 1.5)
        elif score <= -1:
            prediction = "SELL"
            confidence = min(75, 50 + abs(score) * 5)
            target_change = -(2.0 + abs(score) * 1.0)
        else:
            prediction = "HOLD"
            confidence = 50
            target_change = 0.0

        target_price = round(float(current * (1 + target_change / 100)), 2)
        timeframe = "7-14 days"

        return json.dumps({
            "symbol": symbol.upper(),
            "current_price": round(float(current), 4),
            "prediction": prediction,
            "confidence": confidence,
            "score": score,
            "target_price": target_price,
            "target_change_pct": round(target_change, 2),
            "timeframe": timeframe,
            "signals": signals,
            "technicals": {
                "rsi": round(rsi, 2),
                "macd": round(macd, 4),
                "macd_signal": round(signal, 4),
                "sma20": round(float(sma20), 4),
                "sma50": round(float(sma50), 4),
                "bb_upper": round(bb_upper, 4),
                "bb_lower": round(bb_lower, 4),
            }
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


# ─────────────────────────────────────────────
# LANGGRAPH AGENT
# ─────────────────────────────────────────────

tools_list = [get_stock_data, get_crypto_data, get_market_overview, predict_asset]

SYSTEM_PROMPT = """You are QuantAI, an expert financial analyst assistant specializing in stocks and cryptocurrency.

You have access to real-time market data tools:
- get_stock_data: Fetch stock prices, technicals, OHLCV data
- get_crypto_data: Fetch crypto prices, market data from CoinGecko
- get_market_overview: Get major market indices and top crypto overview
- predict_asset: Generate technical analysis predictions

When users ask about stocks or crypto:
1. ALWAYS use tools to get real data first
2. Provide structured analysis with the data
3. Return your response as a JSON object with this structure:
{
  "message": "Your analysis in markdown format",
  "data_type": "stock" | "crypto" | "market" | "prediction" | "chat",
  "asset_data": { ... raw data from tools ... },
  "prediction": { ... prediction data if applicable ... }
}

Always fetch real data using your tools before answering financial questions.
Be concise but insightful. Highlight key signals and risks."""


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def create_agent():
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    llm_with_tools = llm.bind_tools(tools_list)

    def agent_node(state: AgentState):
        messages = state["messages"]
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return "__end__"

    tool_node = ToolNode(tools_list)

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")

    return graph.compile()


# ─────────────────────────────────────────────
# API ENDPOINTS
# ─────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    history: list = []


class QuickRequest(BaseModel):
    symbol: str
    asset_type: str = "stock"


@app.get("/")
def root():
    return {"status": "QuantAI API running", "version": "1.0.0"}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """Main AI chat endpoint powered by LangGraph agent."""
    try:
        agent = create_agent()

        # Build message history
        messages = []
        for h in req.history[-10:]:  # Keep last 10 messages
            if h["role"] == "user":
                messages.append(HumanMessage(content=h["content"]))
            elif h["role"] == "assistant":
                messages.append(AIMessage(content=h["content"]))

        messages.append(HumanMessage(content=req.message))

        result = agent.invoke({"messages": messages})
        last_msg = result["messages"][-1]
        content = last_msg.content

        # Try to parse JSON response
        try:
            # Extract JSON if wrapped in markdown
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            parsed = json.loads(content)
            return parsed
        except:
            return {
                "message": content,
                "data_type": "chat",
                "asset_data": None,
                "prediction": None,
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/market-overview")
async def market_overview():
    """Get market overview without AI."""
    result = get_market_overview.invoke({})
    return json.loads(result)


@app.post("/api/quick-analyze")
async def quick_analyze(req: QuickRequest):
    """Quick stock/crypto analysis without full chat."""
    try:
        if req.asset_type == "crypto":
            data = json.loads(get_crypto_data.invoke({"coin_id": req.symbol}))
        else:
            data = json.loads(get_stock_data.invoke({"symbol": req.symbol}))

        prediction = json.loads(predict_asset.invoke({
            "symbol": req.symbol,
            "asset_type": req.asset_type
        }))

        return {
            "asset_data": data,
            "prediction": prediction,
            "data_type": req.asset_type,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
