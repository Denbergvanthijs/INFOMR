import os

import numpy as np
import pandas as pd
import scipy as sp
from query import (
    get_all_features,
    get_cosine_distance,
    get_euclidean_distance,
    get_features,
    get_manhattan_distance,
)
from scipy.stats import wasserstein_distance
from tqdm.contrib import tzip


def collect_data(fp_features: str, fp_data: str, distance_functions: list):
    df_features = pd.read_csv(fp_features)
    # Preprocess filename column to only keep the filename
    df_features["filename"] = df_features["filename"].apply(lambda x: x.split("/")[-1])

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
        for query_filepath, query_category in tzip(filepaths, categories, desc=f"Querying with {csv_postfix}"):
            # Load feature vector for query shape and db shapes
            query_features = get_features(df_features, query_filepath)
            if query_features is None:
                raise Exception(f"\nNo features found for '{query_filepath}'.")

            # Number of nearest neighbours is equal amount of shapes in that category
            k = df_features[df_features["category"] == query_category].shape[0]

            if distance_function == "knn":
                # Use KDTree in order to perform KNN
                knn_distances, knn_indices = kdtree.query(query_features, k=k)

                # Remove infinite distances
                idx = knn_distances != float("inf")  # Select indices
                knn_distances = knn_distances[idx]  # Remove from both arrays
                knn_indices = knn_indices[idx]

                for indice, distance in zip(knn_indices, knn_distances):
                    match_filename, match_category = filepaths[indice].split("/")[-1], categories[indice]
                    matched_shape = [query_filepath, query_category, match_filename, match_category, distance]
                    matches.append(matched_shape)

                # if len(knn_distances) < k:  # If number of neighbours is less than k
                #     print(f"Only {len(knn_distances)} neighbours found for {query_filepath}.")

            else:  # Use custom distance functions for querying
                # List of distances between query shape and all other shapes in the dataset
                scores = [distance_function(query_features, feature) for feature in features]
                scores = np.array(scores, dtype=float)
                indices = np.arange(len(scores))

                # Remove infinite distances
                idx = scores != float("inf")
                scores = scores[idx]
                indices = indices[idx]

                # Sort scores in ascending order but keep track of the original indices
                sorted_indices, sorted_scores = zip(*sorted(zip(indices, scores), key=lambda x: x[1]))

                # Select k nearest neighbours
                sorted_indices = sorted_indices[:k]
                sorted_scores = sorted_scores[:k]

                # Get the k nearest neighbours
                returned_meshes = [filepaths[i] for i in sorted_indices]

                for returned_mesh, score in zip(returned_meshes, sorted_scores):
                    match_filename = returned_mesh.split("/")[-1]
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
