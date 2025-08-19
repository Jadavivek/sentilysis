import os
import plotly.graph_objects as go
import streamlit as st


models = [
    "albert-base",
    "gated-rnn",
    "hartmann-distilroberta-base",
    "lstm-rnn",
    "rubert-tiny-v2",
    "transformer-megladon"
]

scores = [0.85, 0.92, 0.87, 0.82, 0.84, 0.98]


st.title("Social Wellness Models analysis")


fig = go.Figure(data=[
    go.Bar(name='Model Scores', x=models, y=scores, marker_color='skyblue')
])
fig.update_layout(
    xaxis_title="Models",
    yaxis_title="Scores",
    xaxis_tickangle=-45,
    template="plotly_white"
)

st.plotly_chart(fig)


model_descriptions = {
    "albert-base": "A lightweight transformer model optimized for speed and accuracy.",
    "gated-rnn": "A recurrent neural network with gated mechanisms for better context understanding.",
    "hartmann-distilroberta-base": "A distilled version of RoBERTa fine-tuned for sentiment analysis.",
    "lstm-rnn": "A long short-term memory network designed for sequential data.",
    "rubert-tiny-v2": "A compact transformer model tailored for multilingual sentiment tasks.",
    "transformer-megladon": "A state-of-the-art transformer model with enhanced attention mechanisms."
}

st.subheader("Model Descriptions")
for model in models:
    st.subheader(model)
    st.write(model_descriptions[model])


assets_path = "assets"

gated_rnn_image = os.path.join(assets_path, "gated-rnn.training.stats.png")
megladon_image = os.path.join(assets_path, "megladon.training.stats.png")

st.subheader("Gated-RNN Training Statistics")
st.image(gated_rnn_image, caption="Gated-RNN Training Stats", use_column_width=True)

st.subheader("Transformer Megladon Training Statistics")
st.image(megladon_image, caption="Transformer Megladon Training Stats", use_column_width=True)


# Analysis subsection
st.title("Analysis")

st.write("""
In this section, we provide a detailed analysis of the models used for sentiment analysis. Each model has its own strengths and weaknesses, and the choice of model depends on the specific requirements of the task.
""")

# Detailed comparison of models
analysis = {
    "albert-base": {
        "pros": [
            "Lightweight and efficient.",
            "Optimized for speed and lower memory usage.",
            "Suitable for real-time applications."
        ],
        "cons": [
            "May not capture complex relationships as effectively as larger models."
        ],
        "best_for": "Tasks requiring fast inference and low resource consumption."
    },
    "gated-rnn": {
        "pros": [
            "Effective for sequential data with long-term dependencies.",
            "Gated mechanisms help retain context over longer sequences."
        ],
        "cons": [
            "Slower training compared to transformer-based models.",
            "May struggle with very large datasets."
        ],
        "best_for": "Sentiment analysis in lengthy texts."
    },
    "hartmann-distilroberta-base": {
        "pros": [
            "Balanced performance and computational efficiency.",
            "Faster than full-sized RoBERTa while retaining high accuracy."
        ],
        "cons": [
            "Slightly less accurate than the full-sized RoBERTa model."
        ],
        "best_for": "General-purpose sentiment classification."
    },
    "lstm-rnn": {
        "pros": [
            "Handles sequential data effectively.",
            "Good at mitigating vanishing gradient problems."
        ],
        "cons": [
            "Limited scalability for very long sequences.",
            "Outperformed by transformers in many NLP tasks."
        ],
        "best_for": "Moderately long text sequences."
    },
    "rubert-tiny-v2": {
        "pros": [
            "Compact and efficient.",
            "Multilingual capabilities."
        ],
        "cons": [
            "Lower accuracy compared to larger models.",
            "Limited to specific use cases."
        ],
        "best_for": "Multilingual sentiment analysis tasks."
    },
    "transformer-megladon": {
        "pros": [
            "State-of-the-art performance.",
            "Enhanced attention mechanisms for capturing intricate relationships."
        ],
        "cons": [
            "High computational requirements.",
            "Longer training times."
        ],
        "best_for": "Complex sentiment analysis tasks requiring high accuracy."
    }
}

# Display the analysis
for model, details in analysis.items():
    st.subheader(f"{model} Analysis")
    st.write("**Pros:**")
    st.write("\n".join([f"- {pro}" for pro in details["pros"]]))
    st.write("**Cons:**")
    st.write("\n".join([f"- {con}" for con in details["cons"]]))
    st.write(f"**Best For:** {details['best_for']}")