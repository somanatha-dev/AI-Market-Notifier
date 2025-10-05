"""
market-notifier.py
Orchestrates: fetch latest market data, fetch news, compute sentiment, predict next movement,
and send notification when thresholds or model/sentiment indicate a crash/risk.

Usage:
    python market-notifier.py

Config (edit below or supply env vars):
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
    NEWSAPI_KEY (optional; fallbacks used)
    MODEL_PATH (path to trained model .joblib)
    SYMBOL (e.g. "RELIANCE.NS" for Indian equities via yfinance)
Dependencies (pip):
    pip install requests joblib scikit-learn pandas numpy yfinance nltk feedparser python-telegram-bot pytz schedule
    python -m nltk.downloader vader_lexicon
"""

import os
import time
import datetime as dt
from typing import Optional
import logging

import pytz
import schedule
import joblib
import yfinance as yf
import requests

from predictor import load_model_and_scaler, create_features_from_df, predict_from_model
from news_fetcher import fetch_news_for_symbol
from sentiment_analyzer import sentiment_score_text

# ----- CONFIG -----
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")  # set or fill below
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")      # set or fill below
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", None)
MODEL_PATH = os.getenv("MODEL_PATH", "model.joblib")
SYMBOL = os.getenv("SYMBOL", "TCS.NS")  # change to your ticker (yfinance style)
PREDICTION_LOOKBACK_DAYS = 60
PRICE_DROP_THRESHOLD = 0.03  # 3% intraday drop => notify
TIMEZONE = "Asia/Kolkata"
RUN_HOUR_START = 9   # 9:00
RUN_HOUR_END = 16    # 16:00 (4 PM)
NOTIFY_ON = True
# -------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def send_telegram(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logging.warning("Telegram token/chat not configured. Skipping send.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}
    try:
        r = requests.post(url, json=payload, timeout=10)
        r.raise_for_status()
        logging.info("Sent telegram message.")
    except Exception as e:
        logging.exception("Failed to send telegram message: %s", e)

def is_market_runtime() -> bool:
    tz = pytz.timezone(TIMEZONE)
    now = dt.datetime.now(tz)
    # Weekdays only
    if now.weekday() >= 5:
        return False
    # Time window
    return RUN_HOUR_START <= now.hour < RUN_HOUR_END

def fetch_latest_price(symbol: str):
    ticker = yf.Ticker(symbol)
    # fetch last 5 days to calculate intraday drop
    df = ticker.history(period="5d", interval="15m", auto_adjust=False)
    if df.empty:
        raise RuntimeError("yfinance returned empty data for symbol: " + symbol)
    last_row = df.iloc[-1]
    return float(last_row["Close"]), df

def analyze_and_notify():
    if not is_market_runtime():
        logging.info("Outside market runtime. Skipping this cycle.")
        return

    try:
        price, intraday_df = fetch_latest_price(SYMBOL)
        logging.info("Latest price for %s = %s", SYMBOL, price)
    except Exception:
        logging.exception("Failed to fetch latest price.")
        return

    # intraday drop check (from open of the day)
    try:
        day_df = intraday_df.copy()
        day_df["date"] = day_df.index.tz_convert(TIMEZONE).date
        today = day_df["date"].iloc[-1]
        day_today = day_df[day_df["date"] == today]
        if not day_today.empty:
            open_price = float(day_today.iloc[0]["Open"])
            drop = (open_price - price) / open_price
            logging.info("Open price %s -> current %s drop %.4f", open_price, price, drop)
        else:
            drop = 0.0
    except Exception:
        logging.exception("Failed intraday drop calc.")
        drop = 0.0

    # News + sentiment
    try:
        articles = fetch_news_for_symbol(SYMBOL, api_key=NEWSAPI_KEY, max_results=5)
        combined_text = " ".join([a["title"] + ". " + (a.get("description") or "") for a in articles])
        sentiment = sentiment_score_text(combined_text)
        logging.info("News sentiment: %s", sentiment)
    except Exception:
        logging.exception("News / sentiment failed.")
        articles = []
        sentiment = {"compound": 0.0, "label": "neutral"}

    # Model prediction
    try:
        model, scaler = load_model_and_scaler(MODEL_PATH)
        # get historical OHLCV for lookback
        ticker = yf.Ticker(SYMBOL)
        hist = ticker.history(period=f"{PREDICTION_LOOKBACK_DAYS}d", interval="1d", auto_adjust=False)
        features = create_features_from_df(hist)
        pred = predict_from_model(model, scaler, features)
        # pred is array for each row; use last
        last_pred_proba = model.predict_proba(scaler.transform(features.dropna().iloc[-1:].values.reshape(1, -1)))[0]
        # assuming classes [0=down,1=up]
        prob_up = float(last_pred_proba[1])
        logging.info("Model prob_up = %.3f", prob_up)
    except Exception:
        logging.exception("Model prediction failed.")
        prob_up = 0.5

    # Decision rules
    notify_reasons = []
    if drop >= PRICE_DROP_THRESHOLD:
        notify_reasons.append(f"Intraday drop {drop*100:.2f}% (threshold {PRICE_DROP_THRESHOLD*100:.1f}%)")
    if sentiment["label"] == "negative":
        notify_reasons.append(f"Negative news sentiment (score {sentiment['compound']:.2f})")
    if prob_up < 0.4:
        notify_reasons.append(f"Model signals low probability of up-move (up prob {prob_up:.2f})")

    if notify_reasons:
        msg = f"*ALERT for {SYMBOL}*\nPrice: {price}\n" + "\n".join("- " + r for r in notify_reasons)
        # include top headlines
        if articles:
            msg += "\n\nTop headlines:\n"
            for a in articles[:3]:
                title = a.get("title")
                url = a.get("url")
                msg += f"• [{title}]({url})\n"
        logging.info("Triggering notification: %s", notify_reasons)
        if NOTIFY_ON:
            send_telegram(msg)
    else:
        logging.info("No reasons to notify this cycle.")

def main():
    logging.info("Starting market-notifier for symbol %s", SYMBOL)
    # run every 3 minutes during market runtime
    schedule.every(3).minutes.do(analyze_and_notify)

    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    main()
