import os

import pandas as pd
import streamlit as st
from feature_extraction import calculate_mesh_features
from querying import query

TOP_N = 5
n_iter = 1_000
n_bins = 10
features_path = "./csvs/feature_extraction.csv"
fp_data = "./data_normalized/"

# Retrieve features from the returned meshes
df_features = pd.read_csv(features_path)
# Preprocess filename column to only keep the filename
df_features["filename"] = df_features["filename"].apply(lambda x: x.split("/")[-1])


def show_mesh(fp_mesh):
    command = f'"C:/Program Files/Python38/python.exe" ./code/meshViewer.py --mesh_path {fp_mesh} --visualization_method shade'
    os.system(command)


st.set_page_config(page_title="Rorschach CBSR",
                   page_icon="💡",
                   layout="wide",
                   initial_sidebar_state="expanded")

# Main page
st.title("Rorschach Content-Based Shape Retrieval System")
st.write(f"Upload a shape. The {TOP_N} most similar shapes will be shown below.")

# Sidebar
st.sidebar.title("Configuration")
uploaded_file = st.sidebar.file_uploader("Choose an object...", type=[".obj",])


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

        # Write features to dataframe, use columns from df_features
        df_query = pd.DataFrame([features_query], columns=df_features.columns[2:])

        st.write(f"Total of {len(features_query)} features extracted from query mesh. Features extracted from query mesh:")
        st.dataframe(df_query)
        st.divider()

    st.subheader(f"Top {TOP_N} similar meshes:")
    with st.spinner("Retrieving similar meshes..."):
        # Create an ordered list of meshes retrieved from the dataset based on EMD (with respect to the query mesh)
        returned_meshes = query(features_query, features_path, fp_data)

        # Split list into category and filename
        category, filename = zip(*[mesh.split("/")[-2:] for mesh in returned_meshes])

        # Save to dataframe
        df_returned = pd.DataFrame({"Position": range(1, len(category)+1), "Category": category, "Filename": filename})
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
        # Set new index
        df_features.set_index("Position", inplace=True)
        # Display dataframe
        st.dataframe(df_features)

    st.sidebar.divider()
    # Add button for each returned mesh to display it
    st.sidebar.subheader("Display meshes:")
    st.sidebar.write("Click on the button to display the mesh.")

    for cat, file in zip(df_returned["Category"].tolist(), df_returned["Filename"].tolist()):
        if st.sidebar.button(f"{cat} - {file}"):
            fp_mesh = os.path.join(fp_data, cat, file)
            show_mesh(fp_mesh)
