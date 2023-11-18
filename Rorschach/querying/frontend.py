import json
import os

import numpy as np
import pandas as pd
import streamlit as st
from pymeshlab import MeshSet

from Rorschach.feature_extraction.extraction import (
    calculate_mesh_features,
    normalize_mesh_features,
)
from Rorschach.preprocessing.patch_meshes import clean_mesh
from Rorschach.preprocessing.preprocess import normalize_mesh
from Rorschach.querying.query import (
    get_all_features,
    get_k_closest,
    return_dist_func,
    visualize,
)

# Set numpy random seed for reproducibility
# Otherwise A1 and D1 to D4 will change
np.random.seed(42)

TOP_N = 5
n_iter = 1_000
n_bins = 10
weights = [0.1, 10]  # Weights for elementary and histogram features, respectively
features_path = "./Rorschach/feature_extraction/features_normalized.csv"
fp_data = "./data_normalized/"
fp_normalization_params = "./Rorschach/feature_extraction/normalization_params.json"
normalization_type = "z-score"
ignore_last = n_bins * 5  # n bins for each of the 5 histograms

with open(fp_normalization_params, "r") as f:
    normalization_params = json.load(f)

# Retrieve features from the returned meshes
df_features = pd.read_csv(features_path)
# Preprocess filename column to only keep the filename
df_features["filename"] = df_features["filename"].apply(lambda x: x.split("/")[-1])

# Get all features from the dataset
filepaths, categories, features = get_all_features(features_path)

st.set_page_config(page_title="Rorschach CBSR",
                   page_icon="💡",
                   layout="wide",
                   initial_sidebar_state="expanded")

# Sidebar
st.sidebar.title("Configuration")
TOP_N = st.sidebar.slider("Number of similar meshes to retrieve:", min_value=1, max_value=10, value=TOP_N, step=1)
distance_func = st.sidebar.selectbox("Distance function:", ["EMD only", "Manhattan + EMD", "Euclidean + EMD", "Cosine + EMD", "KNN"])

distance_func = return_dist_func(distance_func)  # Convert string to callable

uploaded_file = st.sidebar.file_uploader("Choose an object...", type=[".obj",])

# Main page
st.title("Rorschach Content-Based Shape Retrieval System")
st.write(f"Upload a shape. The {TOP_N} most similar shapes will be shown below.")


if uploaded_file is not None:
    mesh = uploaded_file.read()

    # Check if folder exists, if not create it
    if not os.path.exists("./frontend_temp"):
        os.mkdir("./frontend_temp")

    with open("./frontend_temp/temp_mesh.obj", "wb") as f:
        f.write(mesh)

    st.divider()
    st.subheader("Query mesh:")

    with st.spinner("Extracting features from query mesh..."):
        v_no, f_no = clean_mesh("./frontend_temp/temp_mesh.obj", "./frontend_temp/temp_mesh_cleaned.obj",
                                cleanMeshes=False, remeshTargVert=10_000)
        meshset = MeshSet()
        meshset.load_new_mesh("./frontend_temp/temp_mesh_cleaned.obj")
        meshset = normalize_mesh(meshset)
        meshset.save_current_mesh("./frontend_temp/temp_mesh_normalized.obj")
        # Load features of query mesh
        features_query = calculate_mesh_features("./frontend_temp/temp_mesh_normalized.obj", "unknown/temp_mesh_normalized.obj",
                                                 "unknown", n_iter=n_iter, n_bins=n_bins)
        features_query = features_query[2:].astype(float)  # Extract only the features, not the filename and category

        # Normalize the feature vector, after removing text but before removing unwanted features
        features_query = normalize_mesh_features(features_query, normalization_params, normalization_type=normalization_type,
                                                 ignore_last=ignore_last)

        # Set NaN, +inf and -inf to 0
        features_query = np.nan_to_num(features_query, nan=0, posinf=0, neginf=0)

        # Write features to dataframe, use columns from df_features
        df_query = pd.DataFrame([features_query], columns=df_features.columns[2:])

        st.write(f"Total of {len(features_query)} features extracted from query mesh. Features extracted from query mesh:")
        st.dataframe(df_query)
        st.divider()

    st.subheader(f"Top {TOP_N} similar meshes:")
    with st.spinner("Retrieving similar meshes..."):
        if distance_func == "KNN":
            # Only import if needed, for speed increase
            from scipy.spatial import KDTree

            kdtree = KDTree(features)  # Build KDTree for KNN
            retrieved_scores, retrieved_indices = kdtree.query(features_query, k=TOP_N)

            # Remove inf distances
            idx = retrieved_scores != float("inf")
            retrieved_scores = retrieved_scores[idx]
            retrieved_indices = retrieved_indices[idx]
        else:
            # Create an ordered list of meshes retrieved from the dataset based on the distance function (with respect to the query mesh)
            retrieved_scores, retrieved_indices = get_k_closest(features_query, features, k=TOP_N,
                                                                distance_function=distance_func, weights=weights)

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
    st.sidebar.write(f"Click on the button to display the query mesh and the top {TOP_N} similar meshes.")

    # Add toggle for showing the wireframe or not
    show_wireframe = st.sidebar.checkbox("Show wireframe", value=False, key="wireframe")

    if st.sidebar.button(f"Visualize the top {TOP_N} meshes"):
        # filename is category + filename
        meshes_to_visualize = fp_data + df_returned["Category"] + "/" + df_returned["Filename"]
        meshes_to_visualize = ["./frontend_temp/temp_mesh_normalized.obj"] + meshes_to_visualize.tolist()
        window_name = f"Rorschach CBSR System - Query mesh and top {TOP_N} similar meshes"

        visualize(meshes_to_visualize, mesh_show_wireframe=show_wireframe, window_name=window_name)
