#!/usr/bin/env python3
"""
market-notifier.py
Main notifier: fetch market snapshot, headlines, sentiment; call predictor;
compute dynamic allocation + tranche schedule; send Telegram HTML message;
persist crash_state.json to prevent duplicate deployments.
"""

import os
import json
import math
import logging
from datetime import datetime, timedelta, time as dtime
import pytz
import requests

# try to import local modules (news, sentiment, predictor)
from news_fetcher import fetch_headlines
from sentiment_analyzer import analyze_texts
from predictor import predict  # this will use trained models if available

# CONFIG
CRASH_SEQUENCE = [20000, 20000, 10000, 20000, 20000, 10000]
FUNDS = [
    "Navi Nifty India Manufacturing Index Fund",
    "Navi Flexi Cap Fund",
    "Navi Nifty Midcap 150 Index Fund",
    "Navi Nifty 50 Index Fund",
]
VIX_THRESHOLD = 20.0
STATE_FILE = "crash_state.json"
IST = pytz.timezone("Asia/Kolkata")
CRASH_TRIGGER_PCT = -3.0
REQUEST_TIMEOUT = 12

# tranche times (IST)
TRANCHE_TIMES = [dtime(hour=10, minute=15), dtime(hour=12, minute=30), dtime(hour=14, minute=50)]
CLOSE_SLOT = dtime(hour=15, minute=10)

# TELEGRAM
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


# ---------------- utility / state ----------------
def now_ist():
    return datetime.now(pytz.utc).astimezone(IST)


def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            logging.exception("Failed to load state; resetting")
    return {"deployed": [False] * len(CRASH_SEQUENCE)}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# ---------------- market fetch (POC using yfinance) ----------------
def fetch_market_data():
    try:
        import yfinance as yf
        ticker = yf.Ticker("^NSEI")
        hist = ticker.history(period="1d", interval="5m")
        if hist.empty:
            return {"error": "no data from yfinance"}
        open_price = float(hist['Close'].iloc[0])
        last = float(hist['Close'].iloc[-1])
        pct = round((last - open_price) / open_price * 100, 2)

        # try India VIX
        vix = None
        try:
            v = yf.Ticker("^INDIAVIX")
            vh = v.history(period="1d", interval="1d")
            if not vh.empty:
                vix = float(vh['Close'].iloc[-1])
        except Exception:
            vix = None

        return {
            "nifty_pct": pct,
            "nifty_price": last,
            "time": now_ist().strftime("%Y-%m-%d %H:%M IST"),
            "vix": vix,
            "fii": None,
            "dii": None,
            "top_movers": []
        }
    except Exception as e:
        logging.exception("fetch_market_data failed")
        return {"error": f"fetch error: {e}"}


# ---------------- allocation / dynamic logic ----------------
def dynamic_allocation(amount, nifty_drop, vix=None, sentiment=0.0):
    """
    Adaptive allocation rule. Returns dict fund->amount (rupees)
    """
    drop = abs(min(0.0, nifty_drop))
    v = vix or 15.0
    s = sentiment

    if drop < 3 and v < 15 and s >= 0:
        weights = [0.20, 0.25, 0.30, 0.25]
    elif drop < 5 and v < 20:
        weights = [0.20, 0.30, 0.20, 0.30]
    elif drop < 7 and v < 25:
        weights = [0.25, 0.25, 0.10, 0.40]
    else:
        weights = [0.25, 0.20, 0.05, 0.50]

    # tilt further away from midcap on strongly negative sentiment
    if s < -0.25:
        weights = [weights[0], weights[1] * 0.95, max(0.05, weights[2] * 0.6), weights[3] + weights[1] * 0.05 + weights[2] * 0.4]
        total = sum(weights)
        weights = [w / total for w in weights]

    per = [round(amount * w) for w in weights]
    diff = amount - sum(per)
    per[0] += diff
    return dict(zip(FUNDS, per))


# ---------------- tranche recommendation ----------------
def recommend_tranches(amount, now_ist_dt, predictor, prefer_equal_if_uncertain=True):
    """
    Returns (schedule_list, recommendation_text)
    schedule_list: [{'time': 'HH:MM IST'|'Immediate (now)' , 'amount': int, 'note': str}, ...]
    """
    prob = predictor.get("probability", 0.0)
    exp_add = predictor.get("expected_additional_pct", 0.0)
    conf = predictor.get("confidence", 0.6)

    if prob >= 0.75 or exp_add >= 1.5:
        weights = [0.50, 0.30, 0.20]
        reason = "High chance of further fall → <b>front-load</b> now"
    elif prob >= 0.45 or exp_add >= 0.6:
        weights = [0.40, 0.30, 0.30]
        reason = "Moderate chance → lean forward"
    else:
        weights = [1 / 3, 1 / 3, 1 / 3]
        reason = "Low probability → equal tranches"

    if conf < 0.35 and prefer_equal_if_uncertain:
        weights = [1 / 3, 1 / 3, 1 / 3]
        reason += " (low confidence → conservative)"

    today = now_ist_dt.date()
    tranche_datetimes = [datetime.combine(today, t).astimezone(IST) for t in TRANCHE_TIMES]
    close_slot_dt = datetime.combine(today, CLOSE_SLOT).astimezone(IST)

    # after last tranche
    if now_ist_dt > tranche_datetimes[-1]:
        # before market close
        if now_ist_dt.time() < dtime(hour=15, minute=30):
            w1 = weights[0]
            w2 = weights[1] + weights[2]
            amounts = [round(amount * w1), round(amount * w2)]
            diff = amount - sum(amounts)
            amounts[0] += diff
            schedule = [
                {"time": "Immediate (now)", "amount": amounts[0], "note": "Execute immediate tranche"},
                {"time": f"{CLOSE_SLOT.strftime('%H:%M')} IST (close)", "amount": amounts[1], "note": "Close-of-day tranche"},
            ]
            recommendation = f"{reason}. Use immediate + close-slot split because it's past last tranche."
            return schedule, recommendation
        else:
            labels = [t.strftime("%H:%M") + " IST" for t in TRANCHE_TIMES]
            amounts = [round(amount * w) for w in weights]
            diff = amount - sum(amounts)
            amounts[0] += diff
            schedule = [{"time": labels[i], "amount": amounts[i], "note": "Next trading day"} for i in range(3)]
            recommendation = f"Market closed → schedule tranches next trading day. {reason}"
            return schedule, recommendation

    # find next upcoming tranche index
    next_idx = 0
    for i, dtm in enumerate(tranche_datetimes):
        if now_ist_dt <= dtm:
            next_idx = i
            break

    # missed previous tranche
    if next_idx > 0 and now_ist_dt > tranche_datetimes[next_idx - 1]:
        missed_amt = round(amount * weights[next_idx - 1])
        rem_amount = amount - missed_amt
        rem_weights = [weights[k] for k in range(next_idx, 3)]
        sum_rw = sum(rem_weights) if sum(rem_weights) > 0 else 1
        remaining_amounts = [round(rem_amount * (rw / sum_rw)) for rw in rem_weights]
        diff = rem_amount - sum(remaining_amounts)
        if remaining_amounts:
            remaining_amounts[0] += diff
        labels = ["Immediate (now)"] + [tranche_datetimes[i].strftime("%H:%M") + " IST" for i in range(next_idx, next_idx + len(remaining_amounts))]
        schedule = [{"time": labels[0], "amount": missed_amt, "note": "Execute missed tranche now"}]
        for idx_r, amt in enumerate(remaining_amounts):
            schedule.append({"time": labels[idx_r + 1], "amount": amt, "note": "Scheduled tranche"})
        recommendation = f"Missed earlier tranche → execute missed slice now, rest at upcoming times. {reason}"
        return schedule, recommendation

    # normal mapping to next 3 tranches
    final_slots = tranche_datetimes[next_idx:next_idx + 3]
    while len(final_slots) < 3:
        final_slots.append(close_slot_dt)
    labels = [dt.strftime("%H:%M") + " IST" for dt in final_slots]
    amounts = [round(amount * w) for w in weights]
    diff = amount - sum(amounts)
    amounts[0] += diff
    schedule = [{"time": labels[i], "amount": amounts[i], "note": "Scheduled tranche" if i > 0 else "Next tranche"} for i in range(3)]
    recommendation = f"{reason}. Execute tranches at the listed times."
    return schedule, recommendation


# ---------------- Telegram sender ----------------
def send_telegram_html(html_text):
    if not BOT_TOKEN or not CHAT_ID:
        logging.warning("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID environment variables; skipping Telegram send.")
        return False
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": html_text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }
    try:
        r = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
        logging.info("Telegram status: %s", r.status_code)
        return r.status_code == 200
    except Exception:
        logging.exception("Telegram send failed")
        return False


# ---------------- main run/check ----------------
def run_check():
    state = load_state()
    data = fetch_market_data()
    if "error" in data:
        send_telegram_html(f"<b>Market fetch error:</b> {data.get('error')}")
        return

    # headlines & sentiment
    headlines = fetch_headlines(limit=6)
    sent = analyze_texts(headlines)  # {'mean_compound', 'items'}
    news_summary = " | ".join([(item.get("title") or "")[:120] + f" ({item.get('compound',0):+.2f})" for item in sent.get("items", [])[:4]]) or "N/A"

    # Try to fetch recent daily df for model if available (best-effort; falls back in predictor)
    recent_df = None
    try:
        import yfinance as yf
        recent_df = yf.download("^NSEI", period="30d", progress=False)
    except Exception:
        recent_df = None

    pred = predict(
        nifty_pct=data.get("nifty_pct", 0.0),
        vix=data.get("vix"),
        news_sentiment=sent.get("mean_compound", 0.0),
        recent_volatility=None,
        recent_market_df=recent_df
    )

    # EOD handling: separate EOD run would be scheduled; keep same format
    now = now_ist()
    if now.hour == 18 and now.minute in (30, 31):
        html = (
            f"<b>📈 EOD Market Summary — {now.strftime('%Y-%m-%d %H:%M IST')}</b>\n"
            f"<b>Market:</b> Nifty {data.get('nifty_pct')}% ({data.get('nifty_price')}) | <b>VIX:</b> {data.get('vix')}\n"
            f"<b>Top news:</b> {news_summary}\n"
            f"<b>Prediction:</b> <b>{pred['probability']*100:.1f}%</b> chance further fall; expected <b>{pred['expected_additional_pct']}%</b> (conf {pred['confidence']*100:.0f}%)\n"
            f"<b>Personal:</b> SIP ₹500×4 = ₹2000 | Crashes used: {sum(state['deployed'])}/{len(state['deployed'])}\n"
        )
        send_telegram_html(html)
        return

    nifty_pct = data.get("nifty_pct")
    if nifty_pct is None:
        logging.warning("No nifty_pct available; skipping run")
        return

    if nifty_pct <= CRASH_TRIGGER_PCT:
        try:
            idx = state["deployed"].index(False)
        except ValueError:
            idx = None

        if idx is not None:
            amount = CRASH_SEQUENCE[idx]
            alloc = dynamic_allocation(amount, nifty_drop=nifty_pct, vix=data.get("vix"), sentiment=sent.get("mean_compound", 0.0))
            state["deployed"][idx] = True

            schedule, rec_text = recommend_tranches(amount, now, pred)

            # compute per-fund per-slot splits
            slot_details = []
            for slot in schedule:
                slot_amt = slot["amount"]
                per_fund = {}
                if amount <= 0:
                    for f in FUNDS:
                        per_fund[f] = 0
                else:
                    for f, f_total in alloc.items():
                        per_fund[f] = round((f_total / float(amount)) * slot_amt)
                    diff = slot_amt - sum(per_fund.values())
                    if diff != 0:
                        first_fund = list(per_fund.keys())[0]
                        per_fund[first_fund] += diff
                slot_details.append({"time": slot["time"], "per_fund": per_fund, "note": slot["note"]})

            # build HTML message
            html = f"<b>⚠️ MARKET DROP ≥ 3% — {data.get('time')}</b>\n\n"
            html += f"<b>📊 Market Snapshot</b>\n"
            html += f"• Nifty: <b>{nifty_pct}%</b> | • VIX: <b>{data.get('vix')}</b>\n"
            html += f"• News Sentiment: <b>{sent.get('mean_compound')}</b>\n"
            html += f"• Prediction: <b>{pred['probability']*100:.1f}%</b> chance further fall; expected <b>{pred['expected_additional_pct']}%</b> (conf {pred['confidence']*100:.0f}%)\n\n"

            html += f"<b>💡 AI-Based Action Plan</b>\n"
            html += f"Crash #{idx+1} → <b>Deploy ₹ {amount}</b>\n\n"
            html += f"<b>Allocation (total)</b>\n"
            for f, v in alloc.items():
                html += f"• {f}: <b>₹{v}</b>\n"
            html += "\n"

            html += f"<b>⏰ Tranche Plan & Per-Fund Amounts</b>\n"
            html += f"<i>{rec_text}</i>\n\n"
            for s in slot_details:
                html += f"<b>{s['time']}</b> — {s['note']}\n"
                for f, v in s['per_fund'].items():
                    html += f"• {f}: <b>₹{v}</b>\n"
                html += "\n"

            html += f"<b>📌 Summary</b>\n"
            html += f"• Crashes used: <b>{sum(state['deployed'])}/{len(state['deployed'])}</b>\n"
            html += f"• SIP continues: <b>₹500 × 4 = ₹2000</b> monthly\n"
            html += f"\n<b>Disclaimer:</b> This is a data-driven heuristic/model signal, not financial advice.\n"

            send_telegram_html(html)
        else:
            logging.info("Crash detected but all crash slots already deployed.")
    else:
        logging.info(f"No crash. Nifty pct {nifty_pct} | news_mean {sent.get('mean_compound')}")

    save_state(state)


if __name__ == "__main__":
    run_check()
