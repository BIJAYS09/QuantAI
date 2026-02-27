"""
Stock & Crypto Prediction API
Built with FastAPI + LangGraph + LangChain
Auth:    JWT (access + refresh tokens) via core/auth.py
Limits:  Multi-tier rate limiting via core/rate_limit.py
Secrets: AWS Secrets Manager / HashiCorp Vault / .env (dev)
"""

import json
import logging
import requests
import numpy as np
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Annotated, Optional, TypedDict, Literal

import yfinance as yf
import pandas as pd

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

# LangChain / LangGraph
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

# Core modules
from core.secrets import secrets, SecretProviderError
from core.config import settings
from core.auth import CurrentUser, get_current_user, get_current_user_optional
from core.rate_limit import limiter, RateLimits, rate_limit_exceeded_handler, user_key
from core.database import init_db, cleanup_expired_tokens
from routers.auth import router as auth_router

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# AGENT SINGLETON
# ─────────────────────────────────────────────────────────────────────────────
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

When users ask about stocks or crypto:
1. ALWAYS use tools to get real data first
2. Return your response as a JSON object:
{
  "message": "Your analysis in markdown format",
  "data_type": "stock" | "crypto" | "market" | "prediction" | "chat",
  "asset_data": { ... raw data from tools ... },
  "prediction": { ... prediction data if applicable ... }
}"""


# ─────────────────────────────────────────────────────────────────────────────
# TOOLS
# ─────────────────────────────────────────────────────────────────────────────

@tool
def get_stock_data(symbol: str, period: str = "3mo") -> str:
    """Fetch historical stock price data and technical indicators.
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

        hist["SMA20"] = hist["Close"].rolling(window=20).mean()
        hist["SMA50"] = hist["Close"].rolling(window=50).mean()
        delta = hist["Close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        hist["RSI"] = 100 - (100 / (1 + gain / loss))
        ema12 = hist["Close"].ewm(span=12).mean()
        ema26 = hist["Close"].ewm(span=26).mean()
        hist["MACD"] = ema12 - ema26
        hist["Signal"] = hist["MACD"].ewm(span=9).mean()

        latest = hist.iloc[-1]
        prev = hist.iloc[-2] if len(hist) > 1 else latest
        week_ago = hist.iloc[-5] if len(hist) >= 5 else hist.iloc[0]
        month_ago = hist.iloc[-22] if len(hist) >= 22 else hist.iloc[0]

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

        return json.dumps({
            "symbol": symbol.upper(), "name": info.get("longName", symbol.upper()),
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
            "market_cap": info.get("marketCap"), "pe_ratio": info.get("trailingPE"),
            "52w_high": info.get("fiftyTwoWeekHigh"), "52w_low": info.get("fiftyTwoWeekLow"),
            "sector": info.get("sector", "N/A"), "chart_data": chart_data,
        })
    except Exception as e:
        logger.error(f"[Tool:get_stock_data] {e}")
        return json.dumps({"error": str(e)})


@tool
def get_crypto_data(coin_id: str) -> str:
    """Fetch cryptocurrency price data from CoinGecko.
    Args:
        coin_id: CoinGecko coin ID like bitcoin, ethereum, solana, cardano
    """
    try:
        resp = requests.get(f"https://api.coingecko.com/api/v3/coins/{coin_id.lower()}",
            params={"localization": False, "tickers": False, "market_data": True,
                    "community_data": False, "developer_data": False}, timeout=10)
        data = resp.json()
        if "error" in data:
            return json.dumps({"error": data["error"]})
        md = data.get("market_data", {})

        chart_resp = requests.get(
            f"https://api.coingecko.com/api/v3/coins/{coin_id.lower()}/market_chart",
            params={"vs_currency": "usd", "days": 60}, timeout=10)
        chart_data = [
            {"date": datetime.fromtimestamp(ts/1000).strftime("%Y-%m-%d"), "close": round(price, 4)}
            for ts, price in chart_resp.json().get("prices", [])
        ]

        return json.dumps({
            "symbol": data.get("symbol", "").upper(), "name": data.get("name", coin_id),
            "current_price": md.get("current_price", {}).get("usd"),
            "market_cap": md.get("market_cap", {}).get("usd"),
            "volume_24h": md.get("total_volume", {}).get("usd"),
            "change_24h": round(md.get("price_change_percentage_24h", 0), 2),
            "change_7d": round(md.get("price_change_percentage_7d", 0), 2),
            "change_30d": round(md.get("price_change_percentage_30d", 0), 2),
            "ath": md.get("ath", {}).get("usd"), "atl": md.get("atl", {}).get("usd"),
            "circulating_supply": md.get("circulating_supply"),
            "total_supply": md.get("total_supply"),
            "market_cap_rank": data.get("market_cap_rank"),
            "description": data.get("description", {}).get("en", "")[:500],
            "chart_data": chart_data,
        })
    except Exception as e:
        logger.error(f"[Tool:get_crypto_data] {e}")
        return json.dumps({"error": str(e)})


@tool
def get_market_overview() -> str:
    """Get an overview of major market indices and top crypto prices."""
    try:
        indices = {"^GSPC": "S&P 500", "^IXIC": "NASDAQ", "^DJI": "Dow Jones", "^VIX": "VIX"}
        result_indices = []
        for sym, name in indices.items():
            try:
                hist = yf.Ticker(sym).history(period="5d")
                if not hist.empty:
                    latest, prev = hist.iloc[-1]["Close"], hist.iloc[-2]["Close"] if len(hist) > 1 else hist.iloc[-1]["Close"]
                    result_indices.append({
                        "symbol": sym, "name": name,
                        "price": round(float(latest), 2),
                        "change_pct": round(float((latest-prev)/prev*100), 2),
                    })
            except Exception:
                pass

        crypto_resp = requests.get("https://api.coingecko.com/api/v3/coins/markets",
            params={"vs_currency": "usd", "order": "market_cap_desc", "per_page": 5,
                    "page": 1, "price_change_percentage": "24h"}, timeout=10)

        return json.dumps({
            "indices": result_indices,
            "top_cryptos": [{
                "id": c["id"], "symbol": c["symbol"].upper(), "name": c["name"],
                "price": c["current_price"],
                "change_24h": round(c.get("price_change_percentage_24h", 0), 2),
                "market_cap": c["market_cap"],
            } for c in crypto_resp.json()],
            "timestamp": datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"[Tool:get_market_overview] {e}")
        return json.dumps({"error": str(e)})


@tool
def predict_asset(symbol: str, asset_type: str = "stock") -> str:
    """Generate a technical analysis prediction for a stock or crypto.
    Args:
        symbol: Ticker or coin name (e.g., AAPL or bitcoin)
        asset_type: 'stock' or 'crypto'
    """
    try:
        if asset_type == "stock":
            hist = yf.Ticker(symbol.upper()).history(period="6mo")
            if hist.empty: return json.dumps({"error": "No data"})
            prices = hist["Close"].values
        else:
            resp = requests.get(f"https://api.coingecko.com/api/v3/coins/{symbol.lower()}/market_chart",
                params={"vs_currency": "usd", "days": 180}, timeout=10)
            prices = np.array([p[1] for p in resp.json().get("prices", [])])

        if len(prices) < 50:
            return json.dumps({"error": "Not enough data"})

        ps = pd.Series(prices)
        sma20, sma50 = ps.rolling(20).mean().iloc[-1], ps.rolling(50).mean().iloc[-1]
        current = prices[-1]
        delta = ps.diff()
        gain, loss = delta.clip(lower=0).rolling(14).mean(), (-delta.clip(upper=0)).rolling(14).mean()
        rsi = float(100 - (100 / (1 + gain.iloc[-1] / loss.iloc[-1])))
        ema12, ema26 = ps.ewm(span=12).mean(), ps.ewm(span=26).mean()
        macd = float(ema12.iloc[-1] - ema26.iloc[-1])
        signal = float((ema12 - ema26).ewm(span=9).mean().iloc[-1])
        bb_mean, bb_std = ps.rolling(20).mean().iloc[-1], ps.rolling(20).std().iloc[-1]
        bb_upper, bb_lower = float(bb_mean + 2*bb_std), float(bb_mean - 2*bb_std)

        score, signals = 0, []
        if current > sma20: score += 1; signals.append("Price above SMA20 (bullish)")
        else: score -= 1; signals.append("Price below SMA20 (bearish)")
        if sma20 > sma50: score += 1; signals.append("Golden cross: SMA20 > SMA50 (bullish)")
        else: score -= 1; signals.append("Death cross: SMA20 < SMA50 (bearish)")
        if rsi < 30: score += 2; signals.append(f"RSI oversold at {rsi:.1f} — strong buy signal")
        elif rsi < 45: score += 1; signals.append(f"RSI at {rsi:.1f} — mildly bullish")
        elif rsi > 70: score -= 2; signals.append(f"RSI overbought at {rsi:.1f} — strong sell signal")
        elif rsi > 55: score -= 1; signals.append(f"RSI at {rsi:.1f} — mildly bearish")
        else: signals.append(f"RSI neutral at {rsi:.1f}")
        if macd > signal: score += 1; signals.append("MACD above signal line (bullish)")
        else: score -= 1; signals.append("MACD below signal line (bearish)")
        if current < bb_lower: score += 1; signals.append("Price below lower Bollinger Band — potential reversal up")
        elif current > bb_upper: score -= 1; signals.append("Price above upper Bollinger Band — potential reversal down")
        if len(prices) >= 6:
            mom = float((prices[-1]-prices[-6])/prices[-6]*100)
            if mom > 3: score += 1; signals.append(f"Strong 5-day momentum: +{mom:.1f}%")
            elif mom < -3: score -= 1; signals.append(f"Weak 5-day momentum: {mom:.1f}%")

        if score >= 3: pred, conf, chg = "STRONG BUY", min(90, 60+score*5), 5.0+score*1.5
        elif score >= 1: pred, conf, chg = "BUY", min(75, 50+score*5), 2.0+score*1.0
        elif score <= -3: pred, conf, chg = "STRONG SELL", min(90, 60+abs(score)*5), -(5.0+abs(score)*1.5)
        elif score <= -1: pred, conf, chg = "SELL", min(75, 50+abs(score)*5), -(2.0+abs(score)*1.0)
        else: pred, conf, chg = "HOLD", 50, 0.0

        return json.dumps({
            "symbol": symbol.upper(), "current_price": round(float(current), 4),
            "prediction": pred, "confidence": conf, "score": score,
            "target_price": round(float(current*(1+chg/100)), 2),
            "target_change_pct": round(chg, 2), "timeframe": "7-14 days",
            "signals": signals,
            "technicals": {
                "rsi": round(rsi,2), "macd": round(macd,4), "macd_signal": round(signal,4),
                "sma20": round(float(sma20),4), "sma50": round(float(sma50),4),
                "bb_upper": round(bb_upper,4), "bb_lower": round(bb_lower,4),
            }
        })
    except Exception as e:
        logger.error(f"[Tool:predict_asset] {e}")
        return json.dumps({"error": str(e)})


tools_list = [get_stock_data, get_crypto_data, get_market_overview, predict_asset]


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


# ─────────────────────────────────────────────────────────────────────────────
# LIFESPAN
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info(f"[Startup] QuantAI — env={settings.app_env}")

    # 1. Validate secrets
    try:
        secrets.initialize()
    except SecretProviderError as e:
        logger.critical(f"[Startup] FATAL — secrets: {e}")
        raise SystemExit(1)

    # 2. Init database
    await init_db(settings.db_path)

    # 3. Build agent singleton
    global _agent
    try:
        _agent = _build_agent()
    except Exception as e:
        logger.critical(f"[Startup] FATAL — agent: {e}")
        raise SystemExit(1)

    logger.info(f"[Startup] Config: {settings.summary()}")
    logger.info("[Startup] Ready ✓")
    logger.info("=" * 60)

    yield

    # Cleanup expired tokens on shutdown
    removed = await cleanup_expired_tokens(settings.db_path)
    if removed:
        logger.info(f"[Shutdown] Cleaned up {removed} expired refresh tokens.")
    logger.info("[Shutdown] Done.")
    _agent = None


# ─────────────────────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="QuantAI — Stock & Crypto Predictor",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None if settings.is_production else "/docs",
    redoc_url=None if settings.is_production else "/redoc",
)

# Rate limiter state on app (required by slowapi)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

# Mount auth router
app.include_router(auth_router)


# ─────────────────────────────────────────────────────────────────────────────
# AUTH INJECTION MIDDLEWARE
# Reads the JWT from the Authorization header and injects user_id into
# request.state so the rate limiter's user_key() can use it.
# ─────────────────────────────────────────────────────────────────────────────

@app.middleware("http")
async def inject_user_state(request: Request, call_next):
    """
    Best-effort JWT extraction for rate limit key injection.
    Does NOT enforce auth — that's done per-endpoint with Depends(get_current_user).
    """
    from fastapi.security.utils import get_authorization_scheme_param
    from core.auth import decode_access_token
    authorization = request.headers.get("Authorization", "")
    scheme, token = get_authorization_scheme_param(authorization)
    if scheme.lower() == "bearer" and token:
        try:
            payload = decode_access_token(token)
            request.state.user_id = payload.get("sub")
        except Exception:
            request.state.user_id = None
    else:
        request.state.user_id = None
    return await call_next(request)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"[Error] {request.url}: {exc}", exc_info=True)
    msg = str(exc) if settings.debug else "An internal error occurred."
    return JSONResponse(status_code=500, content={"detail": msg})


# ─────────────────────────────────────────────────────────────────────────────
# REQUEST MODELS
# ─────────────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    history: list = []


class QuickRequest(BaseModel):
    symbol: str
    asset_type: str = "stock"


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "QuantAI API running", "version": "1.0.0", "env": settings.app_env}


@app.get("/health")
def health():
    agent_ok = _agent is not None
    secrets_health = secrets.health_check()
    return {
        "status": "healthy" if (agent_ok and secrets_health["status"] == "healthy") else "degraded",
        "agent": "ready" if agent_ok else "not_initialized",
        "secrets": secrets_health,
        "config": settings.summary(),
    }


@app.get("/health/secrets/audit")
def secrets_audit(user: CurrentUser = Depends(get_current_user)):
    """Admin only — restrict with require_role("admin") when you add roles."""
    return {"audit_log": secrets.get_audit_log()[-100:]}


# Market overview is public (no auth required) — used for the landing ticker tape
@app.get("/api/market-overview")
@limiter.limit(RateLimits.MARKET_DATA, key_func=user_key)
async def market_overview(request: Request):
    return json.loads(get_market_overview.invoke({}))


# ─────────────────────────────────────────────────────────────────────────────
# PROTECTED ENDPOINTS  (require valid JWT)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/chat")
@limiter.limit(RateLimits.AI_CHAT, key_func=user_key)
async def chat(
    request: Request,
    req: ChatRequest,
    user: CurrentUser = Depends(get_current_user),   # ← 401 if no valid token
):
    """
    AI chat — requires authentication.
    Rate limited to 20/min per user (LLM calls are expensive).
    """
    logger.info(f"[Chat] user={user.user_id} message={req.message[:60]!r}")
    agent = get_agent()

    messages = []
    for h in req.history[-10:]:
        if h["role"] == "user": messages.append(HumanMessage(content=h["content"]))
        elif h["role"] == "assistant": messages.append(AIMessage(content=h["content"]))
    messages.append(HumanMessage(content=req.message))

    result = agent.invoke({"messages": messages})
    content = result["messages"][-1].content

    try:
        if "```json" in content: content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content: content = content.split("```")[1].split("```")[0].strip()
        return json.loads(content)
    except Exception:
        return {"message": content, "data_type": "chat", "asset_data": None, "prediction": None}


@app.post("/api/quick-analyze")
@limiter.limit(RateLimits.MARKET_DATA, key_func=user_key)
async def quick_analyze(
    request: Request,
    req: QuickRequest,
    user: CurrentUser = Depends(get_current_user),   # ← 401 if no valid token
):
    """
    One-click stock/crypto analysis — requires authentication.
    Rate limited to 60/min per user.
    """
    if req.asset_type == "crypto":
        data = json.loads(get_crypto_data.invoke({"coin_id": req.symbol}))
    else:
        data = json.loads(get_stock_data.invoke({"symbol": req.symbol}))
    prediction = json.loads(predict_asset.invoke({"symbol": req.symbol, "asset_type": req.asset_type}))
    return {"asset_data": data, "prediction": prediction, "data_type": req.asset_type}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=settings.is_development)
