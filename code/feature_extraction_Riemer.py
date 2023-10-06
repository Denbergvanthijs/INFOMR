import open3d as o3d
from tqdm import tqdm
import numpy as np
import math
import csv
import os


# Compute the area and volume of a mesh
def compute_area_volume(mesh):
    # Load the mesh with open3d
    mesh.compute_vertex_normals()

    # Compute mesh area and volume
    area = mesh.get_surface_area()
    volume = mesh.get_volume()

    return area, volume


# Compute compactness of a mesh (based on mesh area and volume)
def compute_compactness(area, volume):
    # Calculate compactness
    compactness = (area ** 1.5) / (36 * math.pi * (volume ** 0.5))
    return compactness


# Compute the diameter of a mesh
def compute_diameter(mesh):
    vertices = np.array(mesh.vertices)

    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)

    # Calculate center of the bounding box of mesh
    center = (min_coords + max_coords) / 2

    # Calculate distance from center to all vertices
    distances = np.linalg.norm(vertices - center, axis=1)
    diameter = 2 * np.max(distances)

    return diameter


def extract_features(fp_data, fp_csv_out, n_categories):
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
        for filename in tqdm(os.listdir(fp_cat_in), desc=f"Category: {category}"):
            fp_mesh = os.path.join(fp_cat_in, filename)  # Input mesh file
            print(fp_mesh)

            mesh = o3d.io.read_triangle_mesh(fp_mesh)

            try:
                area, volume = compute_area_volume(mesh)
                compactness = compute_compactness(area, volume)
                diameter = compute_diameter(mesh)
                all_features.append([fp_mesh, area, volume, compactness, diameter])
            except:
                continue
    
    # Save features to CSV file
    with open(fp_csv_out, "w", newline="", encoding="utf-8") as csvfile:
        column_names = ["Filename", "area", "volume", "compactness", "diameter"]
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(column_names)

        for row in all_features:
            csv_writer.writerow(row)



if __name__ == "__main__":
    fp_data = "./data/"
    fp_csv_out = "./csvs/feature_extraction_Riemer.csv"
    n_categories = 1
    extract_features(fp_data, fp_csv_out, n_categories)


    
    

