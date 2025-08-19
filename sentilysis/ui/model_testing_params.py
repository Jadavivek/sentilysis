import streamlit as st
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from collections import Counter
import plotly.express as px

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)

@st.cache_resource
def get_sentiment_analyzer():
    return SentimentIntensityAnalyzer()

def analyze_sentiment_features(text, analyzer):
    vs = analyzer.polarity_scores(text)
    words = word_tokenize(text.lower())
    sentiment_word_counts = Counter()
    intensity_scores = []
    negation_impact = Counter({'positive_negated': 0, 'negative_negated': 0})

    lexicon = analyzer.lexicon

    negation_words = ["not", "n't", "no", "never", "none", "neither", "nor", "hardly", "scarcely", "barely"]

    for i, word in enumerate(words):
        if word in lexicon:
            sentiment_word_counts[word] += 1
            intensity_scores.append(lexicon[word])

            for j in range(max(0, i - 3), i):
                if words[j] in negation_words:
                    if lexicon[word] > 0:
                        negation_impact['positive_negated'] += 1
                    elif lexicon[word] < 0:
                        negation_impact['negative_negated'] += 1
                    break 

    return sentiment_word_counts, intensity_scores, negation_impact

def plot_sentiment_features_plotly(sentiment_word_counts, intensity_scores, negation_impact, text_index, text):
    st.subheader(f"Analysis for Text {text_index + 1}: '{text}'")

    st.subheader("Top 10 Sentiment-Bearing Word Frequencies")
    if sentiment_word_counts:
        most_common_words = sentiment_word_counts.most_common(10)
        words, counts = zip(*most_common_words)
        fig_words = px.bar(x=list(words), y=list(counts),
                             labels={'y': 'Frequency', 'x': 'Word'},
                             title=None)
        fig_words.update_xaxes(tickangle=45)
        st.plotly_chart(fig_words)
    else:
        st.info("No sentiment-bearing words found in the text.")

    st.subheader("Distribution of Sentiment Intensity Scores")
    if intensity_scores:
        fig_intensity = px.histogram(intensity_scores, nbins=20,
                                     labels={'value': 'Intensity Score', 'count': 'Frequency'},
                                     title=None)
        st.plotly_chart(fig_intensity)
    else:
        st.info("No intensity scores available for the text.")

    st.subheader("Impact of Negation on Sentiment")
    negation_labels = list(negation_impact.keys())
    negation_values = list(negation_impact.values())
    fig_negation = px.bar(x=negation_labels, y=negation_values,
                             labels={'y': 'Count', 'x': 'Sentiment Type'},
                             title=None)
    st.plotly_chart(fig_negation)

def main():
    st.title("Feature Analysis for choosing a model")
    text_examples = [
        "This is a very happy and wonderful day!",
        "The food was not good at all, in fact it was terrible.",
        "I am slightly disappointed but still hopeful.",
        "That movie was surprisingly amazing and not bad.",
        "He is incredibly angry and frustrated by the situation."
    ]

    analyzer = get_sentiment_analyzer()

    for i, text in enumerate(text_examples):
        sentiment_word_counts, intensity_scores, negation_impact = analyze_sentiment_features(text, analyzer)

        st.subheader(f"Analysis Results for Text {i+1}: '{text}'")
        st.write("Sentiment Word Counts:", sentiment_word_counts)
        st.write("Intensity Scores:", intensity_scores)
        st.write("Negation Impact:", negation_impact)

        plot_sentiment_features_plotly(sentiment_word_counts, intensity_scores, negation_impact, i, text)
        st.markdown("<br><hr>", unsafe_allow_html=True) # Add a separator

if __name__ == "__main__":
    main()