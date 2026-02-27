# QuantAI — Stock & Crypto Prediction Platform

A real-time market intelligence app powered by **LangGraph**, **LangChain**, **FastAPI**, and a Bloomberg-terminal-inspired UI.

## Architecture

```
stock-predictor/
├── backend/
│   ├── main.py              # FastAPI + LangGraph agent
│   ├── requirements.txt     # Python dependencies
│   └── .env.example         # Environment variables template
└── frontend/
    └── index.html           # Single-file UI (no build required)
```

## Features

- 📈 **Real-time stock data** via `yfinance` (OHLCV, 60-day chart)
- 🪙 **Crypto data** via CoinGecko free API (no key needed)
- 🤖 **LangGraph AI agent** with 4 tools:
  - `get_stock_data` — stock price, volume, technicals
  - `get_crypto_data` — crypto market data + history
  - `get_market_overview` — indices + top 5 cryptos
  - `predict_asset` — RSI/MACD/SMA/Bollinger prediction engine
- 💬 **AI Chat** — conversational market analysis
- 📊 **Technical Analysis** — RSI, MACD, SMA 20/50, Bollinger Bands
- 🎯 **Predictions** — BUY/SELL/HOLD with confidence score & target price

## Setup

### 1. Backend

```bash
cd backend

# Install dependencies
pip install -r requirements.txt

# Copy env and add your OpenAI API key
cp .env.example .env
# Edit .env and set: OPENAI_API_KEY=sk-...

# Run the server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Frontend

Just open `frontend/index.html` in your browser — no build step!

```bash
# Or serve it locally:
cd frontend && python -m http.server 3000
```
Then visit: `http://localhost:3000`

## Usage

### Analyze Tab
- Click any symbol in the sidebar watchlist
- Or type a symbol in the search box (e.g. `GOOGL`, `solana`)

### Market Tab
- Live indices: S&P 500, NASDAQ, Dow Jones, VIX
- Top 5 cryptocurrencies by market cap

### AI Chat Tab
- Ask natural language questions:
  - *"Is NVDA a good buy right now?"*
  - *"What are Bitcoin's technical signals?"*
  - *"Compare Apple and Microsoft"*
  - *"Give me a full market overview"*

## LangGraph Flow

```
User Message
    ↓
[Agent Node] — LLM decides which tool to call
    ↓
[Tool Node] — Executes: get_stock_data / get_crypto_data / predict_asset / market_overview
    ↓
[Agent Node] — Synthesizes tool results into analysis
    ↓
Structured JSON Response → Frontend
```

## Extending

- Add more tools (news sentiment, earnings data, options flow)
- Swap `gpt-4o-mini` for Claude or other models
- Add user authentication + portfolio tracking
- Add WebSocket for live price streaming
- Add backtesting engine
# My_Prediction
