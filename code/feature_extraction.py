import os

import numpy as np
from pymeshlab import MeshSet
from tqdm import tqdm


def compute_a3_histogram(vertices: np.ndarray, n_iter: int = 100, n_bins: int = 10) -> list:
    # A3: angle between 3 random vertices

    angles = []
    for i in range(n_iter):
        # Get 3 random vertices, replace=False means no duplicates
        v1, v2, v3 = vertices[np.random.choice(vertices.shape[0], 3, replace=False)]

        # Compute angle between them
        # angle = compute_angle(v1, v2, v3)
        angle = 10

        # Add angle to histogram
        angles.append(angle)

    # Return N random values temporary as a placeholder
    return [np.random.random() for _ in range(n_bins)]


def extract_features(fp_data: str,  fp_csv_out: str, n_categories: int = 0) -> None:
    meshset = MeshSet()

    categories = next(os.walk(fp_data))[1]
    print(f"Reading {n_categories} categories from {fp_data}...")

    all_features = []
    # Iterate over all classes in the dataset (desklamp, bottle etc.)
    for category in tqdm(categories[:n_categories]):
        fp_cat_in = os.path.join(fp_data, category)  # Input folder

        if not os.path.exists(fp_cat_in):
            print(f"\nThe '{category}' folder does not exist.")
            continue

        # Iterate over all mesh files in current subfolder
        for filename in os.listdir(fp_cat_in):
            fp_mesh = os.path.join(fp_cat_in, filename)  # Input mesh file

            # Load mesh
            meshset.load_new_mesh(fp_mesh)
            mesh = meshset.current_mesh()

            # Get data
            vertices = mesh.vertex_matrix()
            faces = mesh.face_matrix()

            a3 = compute_a3_histogram(vertices)

            # Add filename, category and features to list
            shape_features = np.concatenate(([filename, category], a3))
            all_features.append(shape_features)

    # Save data to CSV
    header = "filename,category," + ",".join([f"a3_{i}" for i in range(10)])

    # Comments='' removes the '#' character from the header
    np.savetxt(fp_csv_out, all_features, delimiter=",", fmt="%s", header=header, comments="")


if __name__ == "__main__":
    fp_data = "./data_normalized/"
    fp_csv_out = "./csvs/feature_extraction.csv"
    n_categories = 1  # len(categories)

    # A3: angle between 3 random vertices
    # D1: distance between barycenter and random vertex
    # D2: distance between 2 random vertices
    # D3: square root of area of triangle given by 3 random vertices
    # D4: cube root of volume of tetrahedron formed by 4 random vertices

    extract_features(fp_data=fp_data, fp_csv_out=fp_csv_out, n_categories=n_categories)
