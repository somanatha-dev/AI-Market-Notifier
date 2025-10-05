#!/usr/bin/env python3
"""
worker.py
Always-on market watcher for Oracle VM deployment.
- Polls market data (yfinance)
- Uses your notifier logic to compute allocation + tranche schedule
- Sends Telegram HTML messages
- Persists crash_state.json
- Exposes /health on port 8080 (localhost by default)
"""

import os
import asyncio
import json
import time
import logging
from datetime import datetime, timezone
import pytz
from aiohttp import web

# try to import helper modules from repo (news, sentiment, predictor, notifier)
try:
    from market_notifier import dynamic_allocation, recommend_tranches, CRASH_SEQUENCE, FUNDS, CRASH_TRIGGER_PCT
    from market_notifier import send_telegram_html as send_telegram_html_main
except Exception:
    # minimal fallback implementations if market_notifier missing/partial
    def dynamic_allocation(amount, nifty_drop, vix=None, sentiment=0.0):
        per = [round(amount/4) for _ in range(4)]
        diff = amount - sum(per)
        per[0] += diff
        return dict(zip(["Mfg","Flexi","Midcap","Nifty50"], per))

    def recommend_tranches(amount, now, predictor):
        amt_each = amount // 3
        labels = ["10:15 IST", "12:30 IST", "14:50 IST"]
        return ([{"time": labels[i], "amount": amt_each if i<2 else amount-2*amt_each, "note": "Scheduled"} for i in range(3)], "Standard equal tranches")

    def send_telegram_html_main(html_text):
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat = os.getenv("TELEGRAM_CHAT_ID")
        if not token or not chat:
            logging.warning("Missing TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID")
            return False
        import requests
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat, "text": html_text, "parse_mode": "HTML", "disable_web_page_preview": True}
        try:
            r = requests.post(url, json=payload, timeout=15)
            logging.info("Telegram send status %s", r.status_code)
            return r.status_code == 200
        except Exception:
            logging.exception("Telegram send failed")
            return False

# predictor import
try:
    from predictor import predict
except Exception:
    def predict(nifty_pct, vix=None, news_sentiment=0.0, recent_volatility=None, recent_market_df=None):
        drop = min(0.0, nifty_pct)
        mag = abs(drop)
        base_prob = min(0.9, mag / 6.0)
        return {"probability": round(base_prob,3), "expected_additional_pct": round(base_prob*0.5*mag,2), "confidence": 0.6, "explanation": {"method": "heuristic"}}

# sentiment (fallback trivial)
try:
    from sentiment_analyzer import analyze_texts
except Exception:
    def analyze_texts(headlines):
        return {"mean_compound": 0.0, "items": []}

# config: default polling/thresholds (override via env)
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "30"))   # secs
ALERT_MIN_INTERVAL = int(os.getenv("ALERT_MIN_INTERVAL", "60"))  # secs
STATE_FILE = "crash_state.json"
IST = pytz.timezone("Asia/Kolkata")

# load persistent state
def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            logging.exception("Failed to load state; resetting")
    return {"deployed": [False] * len(CRASH_SEQUENCE), "last_alert_ts": 0}

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

# market snapshot (sync inside wrapper)
def fetch_market_snapshot_sync():
    try:
        import yfinance as yf
        t = yf.Ticker("^NSEI")
        # 1m interval for near-real-time detection
        hist = t.history(period="1d", interval="1m")
        if hist.empty:
            return None
        open_price = float(hist['Close'].iloc[0])
        last = float(hist['Close'].iloc[-1])
        pct = round((last - open_price) / open_price * 100, 2)
        vix = None
        try:
            v = yf.Ticker("^INDIAVIX")
            vh = v.history(period="1d", interval="1d")
            if not vh.empty:
                vix = float(vh['Close'].iloc[-1])
        except Exception:
            vix = None
        now = datetime.now(timezone.utc).astimezone(IST).strftime("%Y-%m-%d %H:%M:%S IST")
        return {"time": now, "nifty_pct": pct, "nifty_price": last, "vix": vix}
    except Exception:
        logging.exception("Market snapshot failed")
        return None

# safe telegram wrapper
def send_telegram_html_safe(html):
    return send_telegram_html_main(html)

# watcher loop
async def watcher_loop(app):
    logging.info("Watcher started; poll interval %s s", POLL_INTERVAL)
    state = load_state()
    app['state'] = state
    while True:
        try:
            snapshot = await asyncio.get_event_loop().run_in_executor(None, fetch_market_snapshot_sync)
            if snapshot is None:
                logging.debug("No market snapshot")
            else:
                nifty_pct = snapshot.get("nifty_pct")
                logging.info("Snapshot: Nifty %s | VIX %s at %s", nifty_pct, snapshot.get("vix"), snapshot.get("time"))
                if nifty_pct is not None and nifty_pct <= CRASH_TRIGGER_PCT:
                    try:
                        idx = state['deployed'].index(False)
                    except ValueError:
                        idx = None
                    now_dt = datetime.now(IST)
                    if idx is not None:
                        last_ts = state.get("last_alert_ts", 0)
                        if int(time.time()) - int(last_ts) < ALERT_MIN_INTERVAL:
                            logging.info("Alert throttled (recent alert within %s s)", ALERT_MIN_INTERVAL)
                        else:
                            # compute minimal prediction (no heavy news fetch here)
                            pred = predict(nifty_pct=nifty_pct, vix=snapshot.get("vix"), news_sentiment=0.0, recent_market_df=None)
                            amount = CRASH_SEQUENCE[idx]
                            alloc = dynamic_allocation(amount, nifty_drop=nifty_pct, vix=snapshot.get("vix"), sentiment=pred.get("news_sentiment",0.0))
                            schedule, rec_text = recommend_tranches(amount, now_dt, pred)
                            # build message
                            html = f"<b>⚠️ MARKET DROP ≥ 3% — {snapshot.get('time')}</b>\n"
                            html += f"<b>📊</b> Nifty: <b>{nifty_pct}%</b> | VIX: <b>{snapshot.get('vix')}</b>\n\n"
                            html += f"<b>Action:</b> Crash #{idx+1} → <b>Deploy ₹{amount}</b>\n"
                            html += "<b>Allocation (total)</b>\n"
                            for f,v in alloc.items():
                                html += f"• {f}: <b>₹{v}</b>\n"
                            html += "\n<b>Tranche Plan:</b> " + rec_text + "\n\n"
                            for s in schedule:
                                html += f"<b>{s['time']}</b> — {s['note']}\n"
                                # per-fund per-slot info may be attached by main notifier; try to compute proportional split
                                for f,v in alloc.items():
                                    slot_amt = s.get("amount", 0)
                                    per_f = round((v / float(amount)) * slot_amt) if amount>0 else 0
                                    html += f"• {f}: <b>₹{per_f}</b>\n"
                                html += "\n"
                            sent = send_telegram_html_safe(html)
                            if sent:
                                logging.info("Alert sent for crash #%s", idx+1)
                                state['deployed'][idx] = True
                                state['last_alert_ts'] = int(time.time())
                                save_state(state)
                            else:
                                logging.warning("Telegram send failed")
                    else:
                        logging.info("Crash detected but all crash slots deployed")
                else:
                    logging.debug("No crash (nifty %s)", nifty_pct)
        except Exception:
            logging.exception("Watcher loop error")
        await asyncio.sleep(POLL_INTERVAL)

# small health handler
async def health(request):
    st = request.app.get('state', {})
    return web.json_response({"status":"ok","state":st})

def create_app():
    app = web.Application()
    app.router.add_get('/health', health)
    return app

async def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    app = create_app()
    loop = asyncio.get_event_loop()
    loop.create_task(watcher_loop(app))
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '127.0.0.1', int(os.getenv("PORT","8080")))
    await site.start()
    logging.info("Health endpoint serving on 127.0.0.1:8080")
    # keep running
    while True:
        await asyncio.sleep(3600)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Worker shutdown requested")
