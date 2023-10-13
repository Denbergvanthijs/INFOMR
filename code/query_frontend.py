import pandas as pd
import streamlit as st
from feature_extraction import calculate_mesh_features
from querying import query

TOP_N = 5
n_iter = 1000
n_bins = 10
query_path = "./data/Bird/D00089.obj"
features_path = "./csvs/feature_extraction.csv"
fp_data = "./data_normalized/"

st.set_page_config(page_title="RorschachViz CBSR",
                   page_icon="💡",
                   layout="wide",
                   initial_sidebar_state="expanded")

# Main page
st.title("RorschachViz CBSR")
st.write(f"Upload a shape. The {TOP_N} most similar shapes will be shown below.")

# Sidebar
st.sidebar.title("Configuration")
uploaded_file = st.sidebar.file_uploader("Choose an object...", type=[".obj",])


if uploaded_file is not None:
    mesh = uploaded_file.read()

    with open("temp_mesh.obj", 'wb') as f:
        f.write(mesh)

    # st.sidebar.image(shape, caption="Uploaded Image", use_column_width=True)

    st.divider()
    st.subheader(f"Top {TOP_N} most similar shapes:")

    with st.spinner("Retrieving similar shapes..."):
        # To string due to https://stackoverflow.com/a/69581451/10603874
        # df = pd.DataFrame({"Tag": tags, "Probability": probabilities}).round(ROUND_PROBA).astype(str)
        # Query shape/mesh

        # Load features of query mesh
        features_query = calculate_mesh_features("temp_mesh.obj", "unknown/temp_mesh.obj", "unknown", n_iter=n_iter, n_bins=n_bins)
        features_query = features_query[2:].astype(float).tolist()
        print(f"Total of {len(features_query)} features extracted from query mesh.")

        # Create an ordered list of meshes retrieved from the dataset based on EMD (with respect to the query mesh)
        returned_meshes = query(features_query, features_path, fp_data)

        # Write names of first N meshes to screen
        for i in range(TOP_N):
            st.write(f"{i+1}. {returned_meshes[i]}")
