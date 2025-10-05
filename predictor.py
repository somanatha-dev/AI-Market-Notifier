"""
predictor.py

- create_features_from_df(df): produces feature DataFrame (MA, RSI, returns etc.)
- train_model(csv_path, model_out_path): train RandomForestClassifier and save model+scaler.
- load_model_and_scaler(path): returns model, scaler
- predict_from_model(model, scaler, features_df): returns predicted class array for rows

Expect CSV or yfinance DataFrame with columns: Open, High, Low, Close, Volume (datetime index).

Dependencies:
    pip install scikit-learn pandas numpy ta joblib
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import logging

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=(period - 1), adjust=False).mean()
    ema_down = down.ewm(com=(period - 1), adjust=False).mean()
    rs = ema_up / (ema_down + 1e-9)
    return 100 - (100 / (1 + rs))

def create_features_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input: DataFrame with columns Open, High, Low, Close, Volume
    Output: DataFrame of features aligned with df.index (NaNs for first rows)
    """
    df = df.copy().sort_index()
    close = df["Close"]
    features = pd.DataFrame(index=df.index)
    features["ret1"] = close.pct_change(1)
    features["ret2"] = close.pct_change(2)
    features["ma5"] = close.rolling(5).mean()
    features["ma10"] = close.rolling(10).mean()
    features["ma20"] = close.rolling(20).mean()
    features["ma5_ma20_diff"] = features["ma5"] - features["ma20"]
    features["vol_rolling"] = df["Volume"].rolling(10).mean()
    features["rsi14"] = rsi(close, 14)
    # add simple momentum
    features["momentum_5"] = close - close.shift(5)
    # fill NA where safe (we'll drop rows with any NA before training)
    return features

def generate_labels(df: pd.DataFrame, horizon: int = 1) -> pd.Series:
    """
    Binary label: 1 if future close after 'horizon' days > today close else 0
    """
    future = df["Close"].shift(-horizon)
    labels = (future > df["Close"]).astype(int)
    return labels

def train_model(csv_path: str, model_out_path: str = "model.joblib"):
    """
    csv_path: csv with DateTime index and OHLCV columns. If multiple symbols present, supply only one.
    """
    df = pd.read_csv(csv_path, parse_dates=True, index_col=0)
    features = create_features_from_df(df)
    labels = generate_labels(df, horizon=1)

    data = pd.concat([features, labels.rename("label")], axis=1).dropna()
    X = data.drop(columns=["label"])
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train_s, y_train)

    preds = model.predict(X_test_s)
    logging.info("Train complete. Test classification report:\n%s", classification_report(y_test, preds))

    # Save both model and scaler
    joblib.dump({"model": model, "scaler": scaler}, model_out_path)
    logging.info("Saved model to %s", model_out_path)
    return model_out_path

def load_model_and_scaler(path: str = "model.joblib"):
    d = joblib.load(path)
    return d["model"], d["scaler"]

def predict_from_model(model, scaler, features_df: pd.DataFrame):
    """
    features_df: output of create_features_from_df; returns predicted class array aligned to index
    """
    X = features_df.dropna()
    if X.empty:
        return np.array([])
    Xs = scaler.transform(X.values)
    preds = model.predict(Xs)
    # align to index
    return pd.Series(preds, index=X.index)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", help="OHLCV CSV to train on (index as date)", required=False)
    parser.add_argument("--out", help="output model path", default="model.joblib")
    args = parser.parse_args()
    if args.train_csv:
        train_model(args.train_csv, args.out)
    else:
        print("Run with --train-csv <file.csv> to train a model.")
