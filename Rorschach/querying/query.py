import csv
import os

import numpy as np
import open3d as o3d
import pandas as pd
import scipy as sp
from distance_functions import (get_cosine_distance, get_emd,
                                get_euclidean_distance, get_manhattan_distance)
from tqdm import tqdm


def get_features(df_features, fp_mesh) -> list:
    # Obtain features from dataframe based on filename and category

    if not os.path.exists(fp_mesh):
        raise Exception(f"\nThe '{fp_mesh}' file does not exist.")

    # Split on / and obtain category and filename
    category, filename = fp_mesh.split("/")[-2:]

    # Select rows where filename and category match
    df_temp = df_features[df_features["filename"] == filename]
    df_temp = df_temp[df_temp["category"] == category]

    # Drop filename, category, volume, compactness, convexity, and rectangularity columns
    df_temp = df_temp.drop(["filename", "category", "volume", "compactness", "convexity", "rectangularity"], axis=1)
    df_temp = df_temp.astype(float)

    if df_temp.empty:
        return None

    # TODO: add checking for duplicates

    return df_temp.values.tolist()[0]  # Return first row as list


def get_all_features(features_path):
    mesh_paths = []
    categories = []
    features = []

    with open(features_path, newline='') as file:
        csv_reader = csv.reader(file)

        # Skip the first row (header)
        next(csv_reader)

        for row in csv_reader:
            if len(row) >= 1:
                # First element is the mesh path
                mesh_paths.append(row[0])
                # Second element is the category label (Humanoid, Vase, etc.)
                categories.append(row[1])
                # The remainder of the row are the features (excluding 'volume', 'compactness', 'convexity', and 'rectangularity' for now)
                features.append(row[2:3] + row[5:6] + row[7:8] + row[9:])
                # features.append(row[2:])

    return mesh_paths, categories, np.array(features)


def visualize(fp_meshes, width=1280, height=720, mesh_show_wireframe=True) -> None:
    # Function for loading and visualizing meshes

    # Load meshes
    meshes = []
    for i, fp_mesh in enumerate(fp_meshes):
        mesh = o3d.io.read_triangle_mesh(fp_mesh)
        mesh.compute_vertex_normals()

        # Add translation offset
        mesh.translate((i * 0.7 + int(i>0), 0, 0))
        meshes.append(mesh)

    o3d.visualization.draw_geometries(meshes, width=width, height=height, mesh_show_wireframe=mesh_show_wireframe)


# Given a query shape, create an ordered list of meshes from the dataset based on EMD
def query(features_query, df_features, fp_data, distance_function=get_emd) -> list:
    '''Compute the Earth Mover's distance (EMD) between a given query mesh and all meshes in a given dataset. 
    Create an ordered list of meshes based on calculated EMD values. Visualize the query mesh and a specific mesh 
    from the ordered mesh list.

    The mesh list is ordered in ascending order. This means that the mesh with the lowest EMD is located up front, 
    while the mesh with the highest EMD is located at the back.

    Specifically:
    - index 0: the best match based on EMD (i.e. the query mesh itself as EMD = 0).
    - index -1: the worst match based on EMD (i.e. mesh with highest EMD).
    '''
    if not os.path.exists(fp_data):
        raise Exception(f"\nThe '{fp_data}' folder does not exist.")

    # EMD does not work with negative values, so check if any value of the features is negative
    if min(features_query) < 0:
        raise Exception(f"The query mesh features contain negative values. EMD cannot be computed.")

    # Create dict to store pairs of meshes and their Eath Mover's distance to query mesh
    distance_dict = {}

    # Iterate over all classes in the dataset (desklamp, bottle etc.)
    categories = next(os.walk(fp_data))[1]
    for category in tqdm(categories):
        fp_cat_in = os.path.join(fp_data, category)  # Input folder

        if not os.path.exists(fp_cat_in):
            print(f"\nThe '{category}' folder does not exist.")
            continue

        # Iterate over all mesh files in current subfolder
        for filename in tqdm(os.listdir(fp_cat_in), desc=f"Category: {category}"):

            # Obtain full mesh path and load features
            mesh_path = os.path.join(fp_data, category + "/" + filename)
            features_mesh = get_features(df_features, mesh_path)

            # Check if features of current mesh are present
            if features_mesh is None:
                print(f"\nNo features found for '{mesh_path}'. Skipping this mesh.")
                continue

            # NOTE: only use EMD if the minimal possible value of a feature is positive (>= 0)
            # Check if any value of the features is negative
            if min(features_mesh) < 0:
                print(f"\nNegative feature value(s) found for '{mesh_path}'. Skipping this mesh.")
                continue

            # If features of current mesh are present, compute and store EMD to query mesh
            distance_dict[mesh_path] = distance_function(features_query, features_mesh)

    # Sort the mesh filenames based on the distance function
    sorted_distance_dict = sorted(distance_dict.items(), key=lambda item: item[1], reverse=False)

    # Store the sorted filenames
    sorted_meshes, sorted_scores = zip(*sorted_distance_dict)

    return sorted_meshes, sorted_scores


def return_dist_func(selector: str):
    selector_dict = {"EMD": get_emd,
                     "Manhattan": get_manhattan_distance,
                     "Euclidean": get_euclidean_distance,
                     "Cosine": get_cosine_distance}

    return selector_dict[selector]


if __name__ == "__main__":
    # Query shape/mesh
    fp_query = "./data_normalized/Bird/D00089.obj"
    fp_features = "./Rorschach/feature_extraction/features.csv"
    fp_data = "./data_normalized/"
    
    distance_function = get_emd
    # Flag for using KNN instead of custom distance functions
    knn = True

    df_features = pd.read_csv(fp_features)
    # Preprocess filename column to only keep the filename
    df_features["filename"] = df_features["filename"].apply(lambda x: x.split("/")[-1])

    # Load feature vector for query shape and db shapes
    features_query = get_features(df_features, fp_query)
    if features_query is None:
        raise Exception(f"\nNo features found for '{fp_query}'.")
    print(f"Total of {len(features_query)} features extracted from query mesh.")

    # Use KNN for querying
    if knn:
        # Number of nearest neighbours to retrieve
        k = 3

        paths, labels, features = get_all_features(fp_features)
        
        # Use KDTree in order to perform KNN
        kdtree = sp.spatial.KDTree(features)
        knn_distances, knn_indices = kdtree.query(features_query, k=k)
        
        results = [("    " + str(paths[i]) + f" (label='{labels[i]}', distance={dist})") for i, dist in zip(knn_indices, knn_distances)]
        print(f"{k} nearest neighbors for shape {fp_query}:\n",
              "\n".join(results), sep="")

        # Visualize results
        visualize([fp_query] + ["data_normalized/" + paths[i] for i in knn_indices])

    else: # Use custom distance functions for querying
        # Create an ordered list of meshes retrieved from the dataset based on EMD (with respect to the query mesh)
        returned_meshes, sorted_scores = query(features_query, df_features, fp_data, distance_function=distance_function)
        print(f"Number of returned meshes: {len(returned_meshes)}")
        print(f"Best match: {returned_meshes[0]} with EMD: {sorted_scores[0]:3f}")

        # Visualize query mesh and desired mesh from returned mesh list (index 0: best match, index -1: worst match)
        meshes_to_visualize = [fp_query, returned_meshes[0]]
        visualize(meshes_to_visualize)
