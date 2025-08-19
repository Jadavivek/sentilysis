# bhadresh-savani/albert-base-v2-emotion

from base import EmotionAnalyzer, analyze_tweets

def analyze_emotions(tweets):
    analyzer = EmotionAnalyzer("albert-base-v2-emotion")
    return analyze_tweets(tweets, analyzer)