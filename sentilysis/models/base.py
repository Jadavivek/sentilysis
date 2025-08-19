import pandas as pd
from transformers import pipeline

class EmotionAnalyzer:
    def __init__(self, model_name="j-hartmann/emotion-english-distilroberta-base"):
        self.model_name = model_name
        self.pipeline = pipeline(
            "text-classification", model=self.model_name, return_all_scores=True
        )

    def analyze(self, text):
        scores = self.pipeline(text)
        return {score["label"]: score["score"] for score in scores[0]}

    def batch_analyze(self, texts):
        results = []
        for text in texts:
            results.append(self.analyze(text))
        return results

def analyze_tweets(tweets, analyzer):
    texts = [tweet["text"] for tweet in tweets]
    timestamps = [tweet["timestamp"] for tweet in tweets]
    emotions = analyzer.batch_analyze(texts)
    df = pd.DataFrame(emotions)
    df["text"] = texts
    df["timestamp"] = pd.to_datetime(timestamps, errors="coerce")
    df["date"] = df["timestamp"].dt.date
    return df