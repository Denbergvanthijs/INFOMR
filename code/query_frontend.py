import pandas as pd
import streamlit as st

TOP_N = 5
st.set_page_config(page_title="RorschachViz CBIR",
                   page_icon="💡",
                   layout="wide",
                   initial_sidebar_state="expanded")

# Main page
st.title("RorschachViz CBIR")
st.write(f"Upload a shape. The {TOP_N} most similar shapes will be shown below.")

# Sidebar
st.sidebar.title("Configuration")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=[".obj",])


if uploaded_file is not None:
    shape = uploaded_file.read()
    file = {"file": shape}  # Convert to dictionary for requests

    # st.sidebar.image(shape, caption="Uploaded Image", use_column_width=True)

    st.divider()
    st.subheader(f"Top {TOP_N} most similar shapes:")

    with st.spinner("Retrieving similar shapes..."):
        # To string due to https://stackoverflow.com/a/69581451/10603874
        # df = pd.DataFrame({"Tag": tags, "Probability": probabilities}).round(ROUND_PROBA).astype(str)
        print("Sending request...")
