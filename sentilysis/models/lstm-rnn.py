from base import EmotionAnalyzer, analyze_tweets

def analyze_emotions(tweets):
    analyzer = EmotionAnalyzer("Mutasem02/Speech-Emotion-Recognition-SER-using-LSTM-RNN")
    return analyze_tweets(tweets, analyzer)