#!/usr/bin/env python3
"""
predictor.py
Production predictor:
- if trained LightGBM models exist in models/, uses them for inference (probability + expected additional %)
- otherwise falls back to a deterministic heuristic predictor

predict(...) signature:
    predict(nifty_pct, vix=None, news_sentiment=0.0, recent_volatility=None, recent_market_df=None)
returns dict:
    { "probability": float(0..1), "expected_additional_pct": float, "confidence": float(0..1), "explanation": {...} }
"""

import os
import json
import numpy as np

MODEL_DIR = "models"
PROB_PATH = os.path.join(MODEL_DIR, "prob_model.pkl")
REG_PATH = os.path.join(MODEL_DIR, "reg_model.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_cols.json")

# lazy imports
PROB_MODEL = None
REG_MODEL = None
FEATURE_COLS = None

try:
    import joblib
    if os.path.exists(PROB_PATH) and os.path.exists(REG_PATH) and os.path.exists(FEATURES_PATH):
        PROB_MODEL = joblib.load(PROB_PATH)
        REG_MODEL = joblib.load(REG_PATH)
        with open(FEATURES_PATH, "r") as f:
            FEATURE_COLS = json.load(f)
except Exception:
    PROB_MODEL = None
    REG_MODEL = None
    FEATURE_COLS = None

def heuristic_predict(nifty_pct, vix=None, news_sentiment=0.0, recent_volatility=None):
    drop = min(0.0, nifty_pct)
    mag = abs(drop)
    base_prob = min(0.9, mag / 6.0)
    vix_factor = 0.0
    if vix is not None:
        if vix > 30:
            vix_factor = 0.25
        elif vix > 25:
            vix_factor = 0.18
        elif vix > 20:
            vix_factor = 0.12
        elif vix > 15:
            vix_factor = 0.04
    sent_factor = 0.0
    if news_sentiment < -0.3:
        sent_factor = 0.20
    elif news_sentiment < -0.1:
        sent_factor = 0.10
    elif news_sentiment > 0.1:
        sent_factor = -0.05
    vol_factor = 0.0
    confidence = 0.7
    if recent_volatility:
        if recent_volatility > 1.2:
            vol_factor = 0.12
            confidence = 0.48
        elif recent_volatility > 0.6:
            vol_factor = 0.06
            confidence = 0.58
    prob = base_prob + vix_factor + sent_factor + vol_factor
    prob = max(0.01, min(0.99, prob))
    expected_additional = round(prob * max(0.5, mag * 0.5), 2)
    if abs(news_sentiment) < 0.05:
        confidence -= 0.1
    confidence = max(0.2, min(0.95, confidence))
    return {
        "probability": round(prob, 3),
        "expected_additional_pct": expected_additional,
        "confidence": round(confidence, 3),
        "explanation": {"method": "heuristic"}
    }

def _make_feature_row(recent_market_df):
    """
    Make feature row from a pandas DataFrame of daily OHLC indexed by date.
    Mirrors the trainer feature engineering.
    """
    import pandas as pd
    df = recent_market_df.copy()
    if df.empty:
        raise ValueError("recent_market_df empty")
    df = df.sort_index()
    df['ret_1d'] = df['Close'].pct_change() * 100
    for w in [1,3,5,10]:
        df[f'ret_roll_mean_{w}'] = df['ret_1d'].rolling(window=w).mean()
        df[f'ret_roll_std_{w}'] = df['ret_1d'].rolling(window=w).std()
    df['mom_5d'] = df['Close'].pct_change(5) * 100
    df['range_pct'] = (df['High'] - df['Low']) / df['Low'] * 100
    df['open_to_close'] = (df['Close'] - df['Open']) / df['Open'] * 100
    row = df.iloc[-1]
    feat = []
    for c in FEATURE_COLS:
        feat.append(row.get(c, 0.0) if c in row.index else 0.0)
    return np.array(feat).reshape(1, -1)

def predict(nifty_pct, vix=None, news_sentiment=0.0, recent_volatility=None, recent_market_df=None):
    # If models available and we have a recent_market_df, use them
    if PROB_MODEL and REG_MODEL and FEATURE_COLS and recent_market_df is not None:
        try:
            Xrow = _make_feature_row(recent_market_df)
            # LightGBM predict returns probability if trained as classifier with predict; handle both
            prob = None
            try:
                prob = float(PROB_MODEL.predict(Xrow)[0])
            except Exception:
                try:
                    prob = float(PROB_MODEL.predict_proba(Xrow)[:, 1][0])
                except Exception:
                    prob = None
            if prob is None:
                raise RuntimeError("Model predict returned None")
            prob = max(0.0, min(1.0, prob))
            expected_add = float(REG_MODEL.predict(Xrow)[0])
            confidence = max(0.2, min(0.99, abs(prob - 0.5) * 2))
            top_features = {}
            try:
                fi = PROB_MODEL.feature_importance(importance_type="gain")
                if len(fi) == len(FEATURE_COLS):
                    idx_sorted = list(reversed(fi.argsort()[-5:]))
                    top_features = {FEATURE_COLS[i]: int(fi[i]) for i in idx_sorted}
            except Exception:
                top_features = {}
            return {
                "probability": round(prob, 3),
                "expected_additional_pct": round(max(0.0, expected_add), 3),
                "confidence": round(confidence, 3),
                "explanation": {"method": "model", "top_features": top_features}
            }
        except Exception:
            # graceful fallback to heuristic
            return {**heuristic_predict(nifty_pct, vix, news_sentiment, recent_volatility), "explanation": {"method": "fallback"}}
    # fallback
    return heuristic_predict(nifty_pct, vix, news_sentiment, recent_volatility)
