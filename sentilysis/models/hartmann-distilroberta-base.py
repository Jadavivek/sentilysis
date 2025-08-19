from base import EmotionAnalyzer, analyze_tweets

def analyze_emotions(tweets):
    analyzer = EmotionAnalyzer()
    return analyze_tweets(tweets, analyzer)