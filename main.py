"""
Stock & Crypto Prediction API
Built with FastAPI + LangGraph + LangChain
Auth:    JWT (access + refresh tokens) via core/auth.py
Limits:  Multi-tier rate limiting via core/rate_limit.py
Secrets: AWS Secrets Manager / HashiCorp Vault / .env (dev)
"""

import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

# core and routing
from core.secrets import secrets, SecretProviderError
from core.config import settings
from core.auth import CurrentUser, get_current_user, get_current_user_optional
from core.rate_limit import limiter, RateLimits, rate_limit_exceeded_handler, user_key
from core.database import init_db, cleanup_expired_tokens
from routers.auth import router as auth_router

# project modules
from tools import get_stock_data, get_crypto_data, predict_asset, get_market_overview
from agent import init_agent, get_agent
from websocket import manager
from schemas import ChatRequest, QuickRequest

# 
# LOGGING
# 
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s  %(message)s",
)

logger = logging.getLogger(__name__)


# 
# LIFESPAN
# 

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info(f"[Startup] QuantAI  env={settings.app_env}")

    # 1. Validate secrets
    try:
        secrets.initialize()
    except SecretProviderError as e:
        logger.critical(f"[Startup] FATAL  secrets: {e}")
        raise SystemExit(1)

    # 2. Init database
    await init_db(settings.db_path)

    # 3. Build agent singleton
    try:
        init_agent()
    except Exception as e:
        logger.critical(f"[Startup] FATAL  agent: {e}")
        raise SystemExit(1)

    logger.info(f"[Startup] Config: {settings.summary()}")
    logger.info("[Startup] Ready ")
    logger.info("=" * 60)

    yield

    # Cleanup expired tokens on shutdown
    removed = await cleanup_expired_tokens(settings.db_path)
    if removed:
        logger.info(f"[Shutdown] Cleaned up {removed} expired refresh tokens.")
    logger.info("[Shutdown] Done.")


# 
# APP
# 

app = FastAPI(
    title="QuantAI  Stock & Crypto Predictor",
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


# 
# AUTH INJECTION MIDDLEWARE
# 

@app.middleware("http")
async def inject_user_state(request: Request, call_next):
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


# 
# WEBSOCKET ENDPOINTS
# 

@app.websocket("/ws/prices/{symbol}")
async def websocket_price_stream(websocket: WebSocket, symbol: str):
    symbol = symbol.upper()
    await manager.connect(symbol, websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await manager.disconnect(symbol, websocket)
    except Exception as e:
        logger.error(f"[WS] {symbol} error: {e}")
        await manager.disconnect(symbol, websocket)


# 
# PUBLIC ENDPOINTS
# 

@app.get("/")
def root():
    return {"status": "QuantAI API running", "version": "1.0.0", "env": settings.app_env}


@app.get("/health")
def health():
    agent_ok = True
    try:
        _ = get_agent()
    except Exception:
        agent_ok = False
    secrets_health = secrets.health_check()
    return {
        "status": "healthy" if (agent_ok and secrets_health["status"] == "healthy") else "degraded",
        "agent": "ready" if agent_ok else "not_initialized",
        "secrets": secrets_health,
        "config": settings.summary(),
    }


@app.get("/health/secrets/audit")
def secrets_audit(user: CurrentUser = Depends(get_current_user)):
    return {"audit_log": secrets.get_audit_log()[-100:]}


@app.get("/api/market-overview")
@limiter.limit(RateLimits.MARKET_DATA, key_func=user_key)
async def market_overview(request: Request):
    return json.loads(get_market_overview.invoke({}))


# 
# PROTECTED ENDPOINTS
# 

@app.post("/api/chat")
@limiter.limit(RateLimits.AI_CHAT, key_func=user_key)
async def chat(
    request: Request,
    req: ChatRequest,
    user: CurrentUser = Depends(get_current_user),
):
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
    user: CurrentUser = Depends(get_current_user),
):
    symbol = req.symbol.upper()
    if req.asset_type == "crypto":
        data = json.loads(get_crypto_data.invoke({"coin_id": symbol}))
    else:
        data = json.loads(get_stock_data.invoke({"symbol": symbol}))
    prediction = json.loads(predict_asset.invoke({"symbol": symbol, "asset_type": req.asset_type}))
    output = {"asset_data": data, "prediction": prediction, "data_type": req.asset_type}
    if "error" not in data:
        broadcast_msg = {
            "symbol": symbol,
            "price": data.get("current_price"),
            "change_pct": data.get("change_24h") if req.asset_type == "crypto" else data.get("change_1d_pct"),
            "timestamp": datetime.now().isoformat(),
        }
        await manager.broadcast(symbol, broadcast_msg)
    return output


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=settings.is_development)
