import os

import numpy as np
import pandas as pd
import streamlit as st

from Rorschach.feature_extraction.extraction import calculate_mesh_features
from Rorschach.querying.query import get_all_features, get_k_closest, return_dist_func

# Set numpy random seed for reproducibility
# Otherwise A1 and D1 to D4 will change
np.random.seed(42)

TOP_N = 5
n_iter = 1_000
n_bins = 10
features_path = "./Rorschach/feature_extraction/features.csv"
fp_data = "./data_normalized/"

# Retrieve features from the returned meshes
df_features = pd.read_csv(features_path)
# Preprocess filename column to only keep the filename
df_features["filename"] = df_features["filename"].apply(lambda x: x.split("/")[-1])
df_features = df_features.drop(["volume", "compactness", "convexity", "rectangularity"], axis=1)

# Get all features from the dataset
filepaths, categories, features = get_all_features(features_path)


def show_mesh(fp_mesh):
    command = f'"C:/Program Files/Python38/python.exe" ./Rorschach/visualization/meshViewer.py --mesh_path {fp_mesh} --visualization_method shade'
    os.system(command)


st.set_page_config(page_title="Rorschach CBSR",
                   page_icon="💡",
                   layout="wide",
                   initial_sidebar_state="expanded")

# Sidebar
st.sidebar.title("Configuration")
TOP_N = st.sidebar.slider("Number of similar meshes to retrieve:", min_value=1, max_value=10, value=TOP_N, step=1)
distance_func = st.sidebar.selectbox("Distance function:", ("EMD", "Manhattan", "Euclidean", "Cosine"))
distance_func = return_dist_func(distance_func)

uploaded_file = st.sidebar.file_uploader("Choose an object...", type=[".obj",])

# Main page
st.title("Rorschach Content-Based Shape Retrieval System")
st.write(f"Upload a shape. The {TOP_N} most similar shapes will be shown below.")


if uploaded_file is not None:
    mesh = uploaded_file.read()

    with open("temp_mesh.obj", "wb") as f:
        f.write(mesh)

    st.divider()
    st.subheader("Query mesh:")

    with st.spinner("Extracting features from query mesh..."):
        # Load features of query mesh
        features_query = calculate_mesh_features("temp_mesh.obj", "unknown/temp_mesh.obj", "unknown", n_iter=n_iter, n_bins=n_bins)
        features_query = features_query[2:].astype(float).tolist()  # Extract only the features, not the filename and category
        # Ignore volume, compactness, convexity, rectangularity, thus ignore indices 1, 2, 4, 6
        features_query = features_query[:1] + features_query[3:4] + features_query[5:6] + features_query[7:]

        # Write features to dataframe, use columns from df_features
        df_query = pd.DataFrame([features_query], columns=df_features.columns[2:])

        st.write(f"Total of {len(features_query)} features extracted from query mesh. Features extracted from query mesh:")
        st.dataframe(df_query)
        st.divider()

    st.subheader(f"Top {TOP_N} similar meshes:")
    with st.spinner("Retrieving similar meshes..."):
        # Create an ordered list of meshes retrieved from the dataset based on the distance function (with respect to the query mesh)
        retrieved_scores, retrieved_indices = get_k_closest(features_query, features, k=TOP_N, distance_function=distance_func)
        retrieved_meshes = [filepaths[i] for i in retrieved_indices]

        st.write(f"Total of {len(retrieved_meshes)} meshes retrieved. Closest distance: {retrieved_scores[0]:.4f}")

        # Split list into category and filename
        category, filename = zip(*[mesh.split("/")[-2:] for mesh in retrieved_meshes])

        # Save to dataframe
        df_returned = pd.DataFrame({"Position": range(1, len(category)+1),
                                    "Category": category,
                                    "Filename": filename,
                                    "Distance": retrieved_scores})
        df_returned.set_index("Position", inplace=True)
        df_returned = df_returned.head(TOP_N)
        print(df_returned)

        # Select all the returned meshes
        df_features = df_features[df_features["filename"].isin(df_returned["Filename"].tolist())]
        # Sort by the same order as the returned meshes
        df_features = df_features.set_index("filename").loc[df_returned["Filename"].tolist()]
        # Reset index
        df_features.reset_index(inplace=True)
        # Add column with position
        df_features["Position"] = df_returned.index.tolist()
        # Add column with the distance
        df_features["Distance"] = df_returned["Distance"].tolist()
        # Move distance column to the front
        cols = df_features.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        df_features = df_features[cols]
        # Set new index
        df_features.set_index("Position", inplace=True)
        # Display dataframe
        st.dataframe(df_features)

    # Add button for each returned mesh to display it
    st.sidebar.subheader("Display meshes:")
    st.sidebar.write("Click on the button to display the mesh.")

    for cat, file in zip(df_returned["Category"].tolist(), df_returned["Filename"].tolist()):
        if st.sidebar.button(f"{cat} - {file}"):
            fp_mesh = os.path.join(fp_data, cat, file)
            show_mesh(fp_mesh)
