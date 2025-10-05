"""
sentiment_analyzer.py

Simple sentiment using NLTK's VADER lexicon. Returns:
    {"compound": float, "label": "positive"/"neutral"/"negative"}

Dependencies:
    pip install nltk
    python -m nltk.downloader vader_lexicon
"""

from typing import Dict
import logging

from nltk.sentiment.vader import SentimentIntensityAnalyzer

# create singleton analyzer
_analyzer = None
def _get_analyzer():
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentIntensityAnalyzer()
    return _analyzer

def sentiment_score_text(text: str) -> Dict:
    """
    Returns a dict with compound score and label.
    """
    if not text:
        return {"compound": 0.0, "label": "neutral"}
    try:
        sid = _get_analyzer()
        scores = sid.polarity_scores(text)
        compound = float(scores["compound"])
        if compound >= 0.05:
            label = "positive"
        elif compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"
        return {"compound": compound, "label": label}
    except Exception:
        logging.exception("Sentiment analysis failed.")
        return {"compound": 0.0, "label": "neutral"}
