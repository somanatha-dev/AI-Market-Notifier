#!/usr/bin/env python3
"""
sentiment_analyzer.py
Compute VADER sentiment scores for headlines.
Returns: {"mean_compound": float, "items":[{"title","url","compound","pos","neu","neg"}]}
"""
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()

def analyze_texts(headlines):
    if not headlines:
        return {"mean_compound": 0.0, "items": []}
    items = []
    total = 0.0
    for h in headlines:
        text = (h.get("title") or "") + " " + (h.get("summary") or "")
        vs = _analyzer.polarity_scores(text)
        compound = vs["compound"]
        items.append({
            "title": h.get("title"),
            "url": h.get("url"),
            "compound": compound,
            "pos": vs["pos"],
            "neu": vs["neu"],
            "neg": vs["neg"]
        })
        total += compound
    mean = round(total / len(items), 4) if items else 0.0
    return {"mean_compound": mean, "items": items}
