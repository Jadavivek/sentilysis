import os
import streamlit as st

st.title("Reinforcement Learning Pipeline: SARSA + Transformers + Softmax")
st.markdown(
    "This page visualizes the flow from an RL agent (SARSA) through embedding models, ``neural architectures``, and a final Softmax layer."
)

architecture_path = os.path.join("assets", "architecture.svg")
st.image(architecture_path, use_column_width=True)

st.sidebar.header("Components")
st.sidebar.markdown("- **Environment**: The world the agent interacts with.")
st.sidebar.markdown("- **Reward**: Feedback signal guiding learning.")
st.sidebar.markdown("- **SARSA Agent**: On-policy RL algorithm.")
st.sidebar.markdown("- **Models**: Embedding models (BERT, Base Transformer).")
st.sidebar.markdown("- **Neural Architecture**: Transformer and Megladon blocks.")
st.sidebar.markdown("- **Softmax**: Final probability distribution layer.")
