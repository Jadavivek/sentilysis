import streamlit as st 

pages = {
    "Demo" : [
        st.Page("ui/demo.py"),
    ], 
    "Model Building": [
        st.Page("ui/architecture.py"),
        st.Page("ui/model_testing_params.py"),
        st.Page("ui/model_analysis.py"),
    ],
}

pg = st.navigation(pages)
st.set_page_config(
    page_title="Twitter Sentiment and Emotion Analysis",
    page_icon=":bird:",
    layout="wide",
    initial_sidebar_state="expanded",
)
pg.run()