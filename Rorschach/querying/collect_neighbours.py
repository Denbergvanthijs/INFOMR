import os

import numpy as np
import pandas as pd
import scipy as sp
from query import (
    get_all_features,
    get_cosine_distance,
    get_euclidean_distance,
    get_features,
    get_k_closest,
    get_manhattan_distance,
)
from scipy.stats import wasserstein_distance
from tqdm.contrib import tzip


def collect_data(fp_features: str, fp_data: str, distance_functions: list):
    df_features = pd.read_csv(fp_features)
    # Preprocess filename column to only keep the filename
    df_features["filename"] = df_features["filename"].apply(lambda x: x.split("/")[-1])

    # Get number of meshes in each category
    # This is to limit the number of retrieved meshes
    ks = df_features["category"].value_counts().to_dict()

    filepaths, categories, features = get_all_features(fp_features)
    features = np.nan_to_num(features, nan=0, posinf=0, neginf=0)

    filepaths = [os.path.join(fp_data, fp) for fp in filepaths]

    for distance_function in distance_functions:
        # Get a postfix to save each experiment's results in a separate csv file
        if distance_function == "knn":
            csv_postfix = "knn"
        elif distance_function == wasserstein_distance:
            csv_postfix = "emd"
        else:
            csv_postfix = distance_function.__name__.split("_")[1]

        fp_output_csv = f"./Rorschach/evaluation/data/collect_neighbours_{csv_postfix}.csv"

        kdtree = sp.spatial.KDTree(features)  # Build KDTree for KNN only once

        matches = []
        for query_filepath, query_category, query_features in tzip(filepaths, categories, features, desc=f"Querying with {csv_postfix}"):
            k = ks[query_category]  # Number of retrieved meshes is equal amount of meshes in that category

            if distance_function == "knn":
                # Use KDTree in order to perform KNN
                # NOTE: no guarantee that exactly k neighbours will be found, could be less
                knn_distances, knn_indices = kdtree.query(query_features, k=k)

                # Remove infinite distances
                idx = knn_distances != float("inf")  # Select indices
                knn_distances = knn_distances[idx]  # Remove from both arrays
                knn_indices = knn_indices[idx]

                for indice, distance in zip(knn_indices, knn_distances):
                    match_filename, match_category = filepaths[indice].split("/")[-1], categories[indice]
                    matched_mesh = [query_filepath, query_category, match_filename, match_category, distance]
                    matches.append(matched_mesh)

            else:  # Use custom distance functions for querying
                sorted_scores, sorted_indices = get_k_closest(query_features, features, k=k, distance_function=distance_function)

                # Get the k nearest neighbours
                matched_fps = [filepaths[i] for i in sorted_indices]

                for matched_fp, score in zip(matched_fps, sorted_scores):
                    match_filename = matched_fp.split("/")[-1]
                    match_category = df_features[df_features["filename"] == match_filename]["category"].values[0]

                    matched_shape = [query_filepath, query_category, match_filename, match_category, score]
                    matches.append(matched_shape)

        df_matches = pd.DataFrame(matches, columns=["query_filepath", "query_category",
                                                    "match_filepath", "match_category", "distance"])
        df_matches.to_csv(fp_output_csv, index=False)


if __name__ == "__main__":
    # Query shape/mesh
    fp_features = "./Rorschach/feature_extraction/features.csv"
    fp_data = "./data_normalized/"

    # Note that wasserstein_distance == EMD
    distance_functions = ["knn", get_manhattan_distance, get_euclidean_distance, get_cosine_distance, wasserstein_distance]

    df_matches = collect_data(fp_features, fp_data, distance_functions)
