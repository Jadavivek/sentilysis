from base import EmotionAnalyzer, analyze_tweets

def analyze_emotions(tweets):
    analyzer = EmotionAnalyzer("cointegrated/rubert-tiny2-cedr-emotion-detection")
    return analyze_tweets(tweets, analyzer)