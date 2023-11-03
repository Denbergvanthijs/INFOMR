import os

import pandas as pd
import scipy as sp
from tqdm.contrib import tzip
from query import (get_all_features,
                   get_emd,
                   get_features, query, get_euclidean_distance, get_cosine_distance, get_manhattan_distance)

if __name__ == "__main__":
    # Query shape/mesh
    fp_query = "./data_normalized/Bird/D00089.obj"
    fp_features = "./Rorschach/feature_extraction/features.csv"
    fp_data = "./data_normalized/"

    # Flag for using KNN instead of custom distance functions
    distance_function = "knn"
    distance_function = get_manhattan_distance
    # distance_function = get_euclidean_distance
    # distance_function = get_cosine_distance
    # distance_function = get_emd

    if distance_function == "knn":
        csv_postfix = "knn"
    else:
        csv_postfix = distance_function.__name__.split("_")[1]

    fp_output_csv = f"./Rorschach/querying/collect_neighbours_{csv_postfix}.csv"

    df_features = pd.read_csv(fp_features)
    # Preprocess filename column to only keep the filename
    df_features["filename"] = df_features["filename"].apply(lambda x: x.split("/")[-1])

    filepaths, categories, features = get_all_features(fp_features)
    filepaths = [os.path.join(fp_data, fp) for fp in filepaths]
    kdtree = sp.spatial.KDTree(features)

    matches = []
    for filepath, category in tzip(filepaths, categories):
        # Load feature vector for query shape and db shapes
        features_query = get_features(df_features, filepath)
        if features_query is None:
            raise Exception(f"\nNo features found for '{fp_query}'.")

        # Number of nearest neighbours is equal amount of shapes in that category
        k = df_features[df_features["category"] == category].shape[0]

        if distance_function == "knn":
            # Use KDTree in order to perform KNN
            knn_distances, knn_indices = kdtree.query(features_query, k=k)

            # Remove infinite distances
            knn_distances = knn_distances[knn_distances != float("inf")]

            for indice, distance in zip(knn_indices, knn_distances):
                filename = filepaths[indice].split("/")[-1]
                matched_shape = [filepath, category, filename, distance]
                matches.append(matched_shape)

            if len(knn_distances) < k:
                print(f"Only {len(knn_distances)} neighbours found for {filepath}.")

        else:  # Use custom distance functions for querying
            # Create an ordered list of meshes retrieved from the dataset based on EMD (with respect to the query mesh)
            returned_meshes, sorted_scores = query(features_query, df_features, fp_data,
                                                   distance_function=distance_function, enable_tqdm=False)

            # Limit to first k to equal amount of shapes in that category
            returned_meshes = returned_meshes[:k]
            sorted_scores = sorted_scores[:k]

            for returned_mesh, score in zip(returned_meshes, sorted_scores):
                filename = returned_mesh.split("/")[-1]
                matched_shape = [filepath, category, filename, score]
                matches.append(matched_shape)

    df_matches = pd.DataFrame(matches, columns=["query", "category", "match", "distance"])
    df_matches.to_csv(fp_output_csv, index=False)