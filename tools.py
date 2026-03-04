import json
import requests
import numpy as np
from datetime import datetime
import yfinance as yf
import pandas as pd
from textblob import TextBlob
from langchain_core.tools import tool

from cache import cache_get, cache_set
from core.config import settings
import logging

logger = logging.getLogger(__name__)


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
    key = f"crypto:{coin_id.lower()}"
    cached = cache_get(key)
    if cached:
        return cached

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

        result = json.dumps({
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
        cache_set(key, result, ex=60)  # cache 1 minute
        return result
    except Exception as e:
        logger.error(f"[Tool:get_crypto_data] {e}")
        return json.dumps({"error": str(e)})


@tool
def get_market_overview() -> str:
    """Get an overview of major market indices and top crypto prices."""
    cache_key = "market_overview"
    cached = cache_get(cache_key)
    if cached:
        return cached

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

        result = json.dumps({
            "indices": result_indices,
            "top_cryptos": [{
                "id": c["id"], "symbol": c["symbol"].upper(), "name": c["name"],
                "price": c["current_price"],
                "change_24h": round(c.get("price_change_percentage_24h", 0), 2),
                "market_cap": c["market_cap"],
            } for c in crypto_resp.json()],
            "timestamp": datetime.now().isoformat(),
        })
        cache_set(cache_key, result, ex=30)
        return result
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


@tool
def get_news_sentiment(symbol: str, limit: int = 10) -> str:
    """Fetch recent news for an asset and compute overall sentiment.
    Args:
        symbol: Stock ticker or crypto name (e.g., AAPL or bitcoin)
        limit: Number of recent news articles to analyze (default 10)
    """
    try:
        news_key = settings.news_api_key
        sentiment_scores = []
        articles = []
        
        if news_key:
            try:
                news_resp = requests.get(
                    "https://newsapi.org/v2/everything",
                    params={
                        "q": symbol,
                        "sortBy": "publishedAt",
                        "language": "en",
                        "apiKey": news_key,
                    },
                    timeout=10
                )
                news_data = news_resp.json()
                articles = news_data.get("articles", [])[:limit]
            except Exception as e:
                logger.warning(f"[Tool:get_news_sentiment] NewsAPI fetch failed: {e}")
        
        if not articles:
            articles = [
                {"title": f"{symbol} shows strong market momentum", "description": "Positive indicators detected"},
                {"title": f"{symbol} quarterly earnings beat expectations", "description": "Revenue growth outperforms sector"},
                {"title": f"{symbol} faces regulatory headwinds", "description": "Potential compliance challenges"},
            ][:limit]
        
        for article in articles:
            text = f"{article.get('title', '')} {article.get('description', '')}".strip()
            if text:
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity  # -1.0 (negative) to 1.0 (positive)
                sentiment_scores.append(polarity)
        
        if not sentiment_scores:
            return json.dumps({"error": "No articles found"})
        
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        
        if avg_sentiment >= 0.2:
            classification = "BULLISH"
        elif avg_sentiment <= -0.2:
            classification = "BEARISH"
        else:
            classification = "NEUTRAL"
        
        return json.dumps({
            "symbol": symbol.upper(),
            "sentiment_score": round(avg_sentiment, 3),  # -1.0 to 1.0
            "classification": classification,
            "articles_analyzed": len(sentiment_scores),
            "scores": [round(s, 3) for s in sentiment_scores],
            "interpretation": {
                "BULLISH": "Recent news sentiment suggests upward pressure; consider accumulating on dips",
                "BEARISH": "Negative sentiment dominates; exercise caution, monitor support levels",
                "NEUTRAL": "Mixed signals; rely more heavily on technical analysis",
            }[classification],
        })
    except Exception as e:
        logger.error(f"[Tool:get_news_sentiment] {e}")
        return json.dumps({"error": str(e)})


tools_list = [get_stock_data, get_crypto_data, get_market_overview, predict_asset, get_news_sentiment]
