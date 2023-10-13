import csv
import os

import open3d as o3d
from distance_functions import get_emd
from tqdm import tqdm


# Obtain features from CSV file
def get_features(features_path, mesh_path):
    if not os.path.exists(features_path):
        raise Exception(f"\nThe '{features_path}' file does not exist.")
    if not os.path.exists(mesh_path):
        raise Exception(f"\nThe '{mesh_path}' file does not exist.")

    # Split on / and merge last two elements to remove the data/ prefix
    mesh_path = "/".join(mesh_path.split("/")[-2:])

    with open(features_path, newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')

        for i, row in enumerate(csv_reader):
            if i == 0:
                continue
            filepath = row[0]

            if filepath == mesh_path:
                label = row.pop(1)
                row.pop(0)
                features = [float(feature.replace(" ", "")) for feature in row if feature != " "]
                return features

    return None  # No features found for given mesh


# Function for loading and visualizing meshes
def visualize(meshpaths):
    # Load meshes
    meshes = []
    for i, meshpath in enumerate(meshpaths):
        mesh = o3d.io.read_triangle_mesh(meshpath)
        mesh.compute_vertex_normals()

        # Add translation offset
        mesh.translate((i, 0, 0))
        meshes.append(mesh)

    o3d.visualization.draw_geometries(
        meshes,
        width=1280,
        height=720,
        mesh_show_wireframe=True
    )


# Given a query shape, create an ordered list of meshes from the dataset based on EMD
def query(query_path, features_path, fp_data) -> list:
    '''Compute the Earth Mover's distance (EMD) between a given query mesh and all meshes in a given dataset. 
    Create an ordered list of meshes based on calculated EMD values. Visualize the query mesh and a specific mesh 
    from the ordered mesh list.

    The mesh list is ordered in ascending order. This means that the mesh with the lowest EMD is located up front, 
    while the mesh with the highest EMD is located at the back.

    Specifically:
    - index 0: the best match based on EMD (i.e. the query mesh itself as EMD = 0).
    - index -1: the worst match based on EMD (i.e. mesh with highest EMD).
    '''
    if not os.path.exists(query_path):
        raise Exception(f"\nThe '{query_path}' file does not exist.")

    if not os.path.exists(features_path):
        raise Exception(f"\nThe '{features_path}' file does not exist.")

    if not os.path.exists(fp_data):
        raise Exception(f"\nThe '{fp_data}' folder does not exist.")

    # Create dict to store pairs of meshes and their Eath Mover's distance to query mesh
    emd_dict = {}

    # Load features of query mesh
    features_query = get_features(features_path, query_path)
    print(features_query)

    categories = next(os.walk(fp_data))[1]

    # Iterate over all classes in the dataset (desklamp, bottle etc.)
    for category in tqdm(categories):
        fp_cat_in = os.path.join(fp_data, category)  # Input folder

        if not os.path.exists(fp_cat_in):
            print(f"\nThe '{category}' folder does not exist.")
            continue

        # Iterate over all mesh files in current subfolder
        for filename in tqdm(os.listdir(fp_cat_in), desc=f"Category: {category}"):

            # Obtain full mesh path and load features
            mesh_path = os.path.join(fp_data, category + "/" + filename)
            features_mesh = get_features(features_path, mesh_path)

            # If features of current mesh are present, compute and store EMD to query mesh
            if features_mesh is not None:
                emd = get_emd(features_query, features_mesh)
                emd_dict[mesh_path] = emd

                # NOTE: only use EMD if the minimal possible value of a feature is positive (>= 0)

    # Sort the mesh filenames based on EMD
    sorted_emd_dict = sorted(emd_dict.items(), key=lambda item: item[1], reverse=False)

    # Store the sorted filenames
    sorted_meshes = [item[0] for item in sorted_emd_dict]

    return sorted_meshes


if __name__ == "__main__":
    # Query shape/mesh
    query_path = "./data/Bird/D00089.obj"
    features_path = "./csvs/feature_extraction.csv"
    fp_data = "./data_normalized/"

    # Create an ordered list of meshes retrieved from the dataset based on EMD (with respect to the query mesh)
    returned_meshes = query(query_path, features_path, fp_data)
    print(f"Number of returned meshes: {len(returned_meshes)}")
    print(f"Best match: {returned_meshes[0]}")

    # Visualize query mesh and desired mesh from returned mesh list (index 0: best match, index -1: worst match)
    meshes_to_visualize = [query_path, returned_meshes[0]]
    visualize(meshes_to_visualize)
