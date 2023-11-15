import os

import numpy as np
import open3d as o3d
import pandas as pd
import scipy as sp
from distance_functions import (
    get_cosine_distance,
    get_euclidean_distance,
    get_manhattan_distance,
)
from scipy.stats import wasserstein_distance


def get_features(df_features, fp_mesh) -> list:
    # Obtain features from dataframe based on filename and category
    fp_mesh = os.path.normpath(fp_mesh)  # To make /, \ and \\ work

    if not os.path.exists(fp_mesh):
        raise Exception(f"\nThe '{fp_mesh}' file does not exist.")

    # Split on / and obtain category and filename
    category, filename = fp_mesh.split(os.sep)[-2:]  # Split normalised path on seperator that is dependent on OS

    # Select rows where filename and category match
    df_temp = df_features[df_features["filename"] == filename]
    df_temp = df_temp[df_temp["category"] == category]

    # Drop filename, category
    df_temp = df_temp.drop(["filename", "category"], axis=1)
    df_temp = df_temp.astype(float)

    if df_temp.empty:
        return None

    # TODO: add checking for duplicates

    return df_temp.values[0]  # Return first row as list


def get_all_features(features_path):
    if not os.path.exists(features_path):
        raise Exception(f"\nThe '{features_path}' file does not exist.")

    df = pd.read_csv(features_path)

    mesh_paths = df["filename"].values
    categories = df["category"].values
    features = df.drop(["filename", "category"], axis=1).astype(float).values

    return mesh_paths, categories, np.array(features).astype(float)


# Custom distance function
def compute_distance(query_features, current_features, distance_function, weights=[0.1, 10],
                     split_idx: int = 7, n_hists: int = 5) -> float:
    # Compute distance over all elementary features
    elementary_distance = distance_function(query_features[:split_idx], current_features[:split_idx])

    # Split features into 5 equal groups, assuming elementary features are first and properly indexed
    histograms_query = np.split(query_features[split_idx:], n_hists)
    histograms_current = np.split(current_features[split_idx:], n_hists)

    # Compute EMD over all shape property features (histograms)
    emd_distances = sum([wasserstein_distance(histograms_query[i], histograms_current[i]) for i in range(5)])

    return elementary_distance * weights[0] + emd_distances * weights[1]


def visualize(fp_meshes: str, width: int = 1280, height: int = 720,
              mesh_show_wireframe: bool = True, mesh_show_back_face: bool = True, window_name: str = "Rorschach CBSR System") -> None:
    # Function for loading and visualizing meshes

    # Load meshes
    meshes = []
    for i, fp_mesh in enumerate(fp_meshes):
        mesh = o3d.io.read_triangle_mesh(fp_mesh)
        mesh.compute_vertex_normals()

        # Add translation offset
        mesh.translate((i * 0.7 + int(i > 0), 0, 0))
        meshes.append(mesh)

    o3d.visualization.draw_geometries(meshes, width=width, height=height,
                                      window_name=window_name, mesh_show_back_face=mesh_show_back_face)


def get_k_closest(query_features: np.ndarray, features: np.ndarray, k: int, distance_function: callable) -> tuple:
    """Retrieves the k closest neighbours of a query mesh from a dataset of meshes

    :param query_features: List of features of the query mesh
    :type query_features: np.ndarray
    :param features: List of list of features of all meshes in the dataset
    :type features: np.ndarray
    :param k: Number of nearest neighbours to retrieve
    :type k: int
    :param distance_function: Distance function to use
    :type distance_function: callable

    :return: Tuple of lists of scores and indices of the k nearest neighbours
    :rtype: tuple
    """
    # List of distances between query mesh and all other meshes in the dataset
    scores = [compute_distance(query_features, feature, distance_function) for feature in features]
    scores = np.array(scores, dtype=float)
    indices = np.arange(len(scores))

    # Remove infinite distances
    idx = scores != float("inf")
    scores = scores[idx]
    indices = indices[idx]

    # Sort scores in ascending order but keep track of the original indices
    sorted_indices = np.argsort(scores)[:k]
    sorted_scores = scores[sorted_indices]  # Select from k scores

    return sorted_scores, sorted_indices


def return_dist_func(selector: str):
    selector_dict = {"Manhattan": get_manhattan_distance,
                     "Euclidean": get_euclidean_distance,
                     "Cosine": get_cosine_distance,
                     "EMD": wasserstein_distance,
                     "KNN": "KNN"}

    return selector_dict[selector]


if __name__ == "__main__":
    # Query shape/mesh
    fp_query = "./data_normalized/Bird/D00089.obj"
    fp_features = "./Rorschach/feature_extraction/features_normalized.csv"
    fp_data = "./data_normalized/"
    k = 5  # Number of nearest neighbours to retrieve

    distance_function = get_euclidean_distance
    # Flag for using KNN instead of custom distance functions
    knn = False

    # Retrieve features from the returned meshes
    df_features = pd.read_csv(fp_features)
    # Preprocess filename column to only keep the filename
    df_features["filename"] = df_features["filename"].apply(lambda x: x.split("/")[-1])

    features_query = get_features(df_features, fp_query)  # First get features, before dropping columns

    # Get all features from the dataset
    filepaths, categories, features = get_all_features(fp_features)

    # Load feature vector for query shape and db shapes
    if features_query is None:
        raise Exception(f"\nNo features found for '{fp_query}'.")
    print(f"Total of {len(features_query)} features extracted from query mesh.")

    # Use KNN for querying
    if knn:
        # Use KDTree in order to perform KNN
        kdtree = sp.spatial.KDTree(features)
        knn_distances, knn_indices = kdtree.query(features_query, k=k)

        results = [("    " + str(filepaths[i]) + f" (label='{categories[i]}', distance={dist})")
                   for i, dist in zip(knn_indices, knn_distances)]
        print(f"{k} nearest neighbors for shape {fp_query}:\n",
              "\n".join(results), sep="")

        # Visualize results
        meshes_to_visualize = [fp_query] + ["data_normalized/" + filepaths[i] for i in knn_indices]

    else:  # Use custom distance functions for querying
        # Create an ordered list of meshes retrieved from the dataset based on EMD (with respect to the query mesh)
        sorted_scores, sorted_indices = get_k_closest(features_query, features, k=k, distance_function=distance_function)
        returned_meshes = ["./data_normalized/" + filepaths[i] for i in sorted_indices]

        print(f"Number of returned meshes: {len(returned_meshes)}")
        print(f"Best match: {returned_meshes[0]} with distance: {sorted_scores[0]:3f}")

        # Visualize query mesh and desired mesh from returned mesh list (index 0: best match, index -1: worst match)
        meshes_to_visualize = [fp_query] + returned_meshes

    visualize(meshes_to_visualize)
