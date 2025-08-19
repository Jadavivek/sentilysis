import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
from integrations.x import scrape_tweets
from models.model import analyze_emotions_with_llm, EmotionResult


def generate_wordcloud(tweets):
    text = " ".join(tweet["text"] for tweet in tweets)
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(
        text
    )
    return wordcloud


def main():
    st.title("[DEMO]: Social Wellness Analyzer")

    username = st.text_input("Enter Twitter username:", "elonmusk")
    deep_search_factor = st.slider(
        "Amount of searching you want to perform, the more the more thorough the search",
        1,
        20,
        5,
    )

    if st.button("Analyze Tweets"):
        with st.spinner("Fetching tweets..."):
            tweets = scrape_tweets(username, deep_search_factor)

        if not tweets:
            st.error("No tweets found or failed to fetch tweets.")
            return

        st.success(f"Fetched {len(tweets)} tweets!")

        st.subheader("Word Cloud")
        wordcloud = generate_wordcloud(tweets)
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

        df = pd.DataFrame(tweets)
        st.subheader("[Debug] Tweet View")
        with st.expander("Show Raw Tweets with Timestamp"):
            new_df = df.copy()
            new_df["summary"] = new_df["text"]
            st.write(df)

        st.subheader("Emotion Analysis")

        emotion_df = analyze_emotions_with_llm(tweets)

        st.subheader("Wellness Score")
        emotion_df["wellness_score"] = emotion_df.apply(
            lambda row: EmotionResult(
                joy=row["joy"],
                sadness=row["sadness"],
                anger=row["anger"],
                fear=row["fear"],
                surprise=row["surprise"],
                neutral=row["neutral"],
            ).wellness_score(),
            axis=1,
        )
        average_wellness_score = emotion_df["wellness_score"].mean()
        st.metric(label="Overall Wellness Score", value=f"{average_wellness_score:.2f}")

        st.subheader("Average Emotion by Day")
        daily_emotion = emotion_df.groupby("date").mean(numeric_only=True)
        st.line_chart(daily_emotion)


if __name__ == "__main__":
    main()
