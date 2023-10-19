import itertools
import math
import os

import numpy as np
import open3d as o3d
from pymeshlab import MeshSet
from scipy.spatial import ConvexHull
from tqdm import tqdm


# Compute the area and volume of a mesh
def compute_area_volume(fp_mesh):
    # Load the mesh with open3d
    mesh = o3d.io.read_triangle_mesh(fp_mesh)
    mesh.compute_vertex_normals()

    # Compute mesh area and volume
    area = mesh.get_surface_area()

    if mesh.is_watertight():
        volume = mesh.get_volume()
    else:
        volume = -1

    return area, volume


# Compute compactness of a mesh (based on mesh area and volume)
def compute_compactness(area, volume):
    # Return -1 if volume is invalid due to previous error
    if volume == -1:
        return -1

    compactness = (area ** 1.5) / (36 * math.pi * (volume ** 0.5))
    return compactness


# Compute the diameter of a mesh
def compute_diameter(fp_mesh):
    # Load the mesh with open3d
    mesh = o3d.io.read_triangle_mesh(fp_mesh)
    vertices = np.array(mesh.vertices)

    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)

    # Calculate center of the bounding box of mesh
    center = (min_coords + max_coords) / 2

    # Calculate distance from center to all vertices
    distances = np.linalg.norm(vertices - center, axis=1)
    diameter = 2 * np.max(distances)

    return diameter


# Compute convexity of a mesh (mesh volume over convex hull volume)
def compute_convexity(vertices, mesh_volume):
    # Return -1 if there are not enough vertices or not minimal 4 unique x coordinates
    if vertices.shape[0] < 4 or len(np.unique(vertices[:, 0])) < 4:
        return -1

    # Calculate convex hull and corresponding volume
    convex_hull = ConvexHull(vertices)
    convex_hull_volume = convex_hull.volume

    # Divide mesh volume by convex hull volume
    convexity = mesh_volume / convex_hull_volume
    return convexity


# Compute eccentricity of a mesh (ratio of largest to smallest eigenvalues of covariance matrix)
def compute_eccentricity(vertices):
    # Compute covariance matrix and corresponding eigenvalues
    cov_matrix = np.cov(vertices, rowvar=False)
    eigenvalues, _ = np.linalg.eig(cov_matrix)

    if min(eigenvalues) == 0:  # Avoid division by zero
        return -1

    # Compute and return eccentricity
    eccentricity = max(eigenvalues) / min(eigenvalues)
    return eccentricity


# Compute 3D rectangularity of mesh (shape volume divided by OBB volume)
def compute_rectangularity(mesh_path, mesh_volume):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()

    # Obtain oriented bounding box (OBB)
    obb = o3d.geometry.OrientedBoundingBox(mesh)
    obb = obb.get_oriented_bounding_box()

    # Obtain OBB volume
    obb_volume = obb.volume()

    # Calculate 3D rectangularity based on mesh volume and OBB volume
    rectangularity = mesh_volume / obb_volume
    return rectangularity


def compute_angle_3D(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray) -> float:
    # Based on: https://stackoverflow.com/a/35178910/10603874
    v2v1 = v1 - v2  # Normalized vectors
    v2v3 = v3 - v2

    cosine_angle = np.dot(v2v1, v2v3) / (np.linalg.norm(v2v1) * np.linalg.norm(v2v3))
    angle = np.arccos(cosine_angle)  # Angle in radians

    return np.degrees(angle)  # Angle in degrees from 0 to 180


def compute_a3_hist(vertices: np.ndarray, n_iter: int = 1_000, n_bins: int = 10) -> list:
    # A3: angle between 3 random vertices

    angles = []
    for _ in range(n_iter):
        # Get 3 random vertices, replace=False means no duplicates
        v1, v2, v3 = vertices[np.random.choice(vertices.shape[0], 3, replace=False)]

        angle = compute_angle_3D(v1, v2, v3)  # Compute angle between them
        angles.append(angle)  # Add angle to histogram

    hist, _ = np.histogram(angles, bins=n_bins, range=(0, 180))  # Create histogram
    hist = hist / np.sum(hist)  # Normalize histogram

    return hist


def compute_d1_hist(vertices: np.ndarray, barycenter: np.ndarray, n_iter: int = 1_000, n_bins: int = 10) -> list:
    # D1: distance between barycenter and random vertex

    # Use replace=False to avoid duplicates, only when n_iter < vertices.shape[0]
    if n_iter < vertices.shape[0]:
        indices = np.random.choice(vertices.shape[0], n_iter, replace=False)
    else:  # If n_iter >= vertices.shape[0], use all vertices but not more than vertices.shape[0]
        indices = np.arange(vertices.shape[0])

    distances = np.linalg.norm(vertices[indices] - barycenter, axis=1)

    hist, _ = np.histogram(distances, bins=n_bins, range=(0, 1))  # Create histogram
    hist = hist / np.sum(hist)  # Normalize histogram

    return hist


def compute_d2_hist(vertices: np.ndarray, n_iter: int = 1_000, n_bins: int = 10, guarantee_unique_pairs: bool = False) -> list:
    # D2: distance between 2 random vertices

    if guarantee_unique_pairs:
        # Get the indices of all possible pairs of vertices
        indices = np.array(list(itertools.combinations(range(vertices.shape[0]), 2)))

        # Use replace=False to avoid duplicates, only when n_iter < indices.shape[0]
        # Since not all meshes have enough unique vertices, we need to check this
        if n_iter < indices.shape[0]:
            indices = indices[np.random.choice(indices.shape[0], n_iter, replace=False)]

        # Compute distance between each pair of vertices
        distances = np.linalg.norm(vertices[indices[:, 0]] - vertices[indices[:, 1]], axis=1)

    else:
        distances = []
        for _ in range(n_iter):
            # Get 2 random vertices, replace=False means v1 and v2 are not the same
            # However, this does not guarantee that the next pair of vertices is not the same
            v1, v2 = vertices[np.random.choice(vertices.shape[0], 2, replace=False)]

            distance = np.linalg.norm(v1 - v2)  # Compute distance between them
            distances.append(distance)

    hist, _ = np.histogram(distances, bins=n_bins, range=(0, 1))  # Create histogram
    hist = hist / np.sum(hist)  # Normalize histogram

    return hist


def compute_d3_hist(vertices: np.ndarray, n_iter: int = 1_000, n_bins: int = 10) -> list:
    # D3: square root of area of triangle given by 3 random vertices

    areas = []
    for _ in range(n_iter):
        # Get 3 random vertices, replace=False means no duplicates
        v1, v2, v3 = vertices[np.random.choice(vertices.shape[0], 3, replace=False)]

        # Compute area of triangle given by 3 vertices
        # Based on https://math.stackexchange.com/a/128999
        area = np.linalg.norm(np.cross(v1 * v2, v1 * v3)) / 2

        areas.append(area)

    # Square root of area
    areas = np.sqrt(areas)

    hist, _ = np.histogram(areas, bins=n_bins, range=(0, max(areas)))  # Create histogram
    hist = hist / np.sum(hist)  # Normalize histogram

    return hist


def compute_d4_hist(vertices: np.ndarray, n_iter: int = 1_000, n_bins: int = 10) -> list:
    # D4: cube root of volume of tetrahedron formed by 4 random vertices

    if vertices.shape[0] < 4:
        return np.zeros(n_bins)  # Return empty histogram since there are not enough vertices

    volumes = []
    for _ in range(n_iter):
        # Get 4 random vertices, replace=False means no duplicates
        v1, v2, v3, v4 = vertices[np.random.choice(vertices.shape[0], 4, replace=False)]

        # Compute volume of tetrahedron formed by 4 vertices
        # Based on https://en.wikipedia.org/wiki/Tetrahedron
        volume = np.linalg.norm(np.linalg.det([v1 - v4, v2 - v4, v3 - v4])) / 6

        # Alternative solution using cross and dot product. TODO: investigate which one is faster
        # volume = np.linalg.norm(np.dot(v1 - v4, np.cross(v2 - v4, v3 - v4))) / 6
        # assert np.isclose(volume, volume2)

        volumes.append(volume)

    # Cube root of volumecd
    volumes = np.cbrt(volumes)

    hist, _ = np.histogram(volumes, bins=n_bins, range=(0, volumes.max()))  # Create histogram
    hist = hist / np.sum(hist)  # Normalize histogram

    return hist


def calculate_mesh_features(fp_mesh: str, full_filename: str, category: str, n_iter: int = 1_000, n_bins: int = 10) -> None:
    meshset = MeshSet()
    meshset.load_new_mesh(fp_mesh)
    mesh = meshset.current_mesh()

    # Get data
    vertices = mesh.vertex_matrix()
    measures = meshset.get_geometric_measures()  # Dictionary with geometric measures
    barycenter = measures["barycenter"]

    # Compute global features
    area, volume = compute_area_volume(fp_mesh)
    compactness = compute_compactness(area, volume)
    diameter = compute_diameter(fp_mesh)
    convexity = compute_convexity(vertices, volume)
    eccentricity = compute_eccentricity(vertices)
    rectangularity = compute_rectangularity(fp_mesh, volume)

    # Store global features as well as filename and category
    global_features = np.array([full_filename, category, area, volume, compactness, diameter, convexity, eccentricity, rectangularity])

    # Compute shape property features
    a3 = compute_a3_hist(vertices, n_iter=n_iter, n_bins=n_bins).round(3)  # Round to 3 decimal places
    d1 = compute_d1_hist(vertices, barycenter, n_iter=n_iter, n_bins=n_bins).round(3)  # for floating point errors like 0.300004
    d2 = compute_d2_hist(vertices, n_iter=n_iter, n_bins=n_bins).round(3)
    d3 = compute_d3_hist(vertices, n_iter=n_iter, n_bins=n_bins).round(3)
    d4 = compute_d4_hist(vertices, n_iter=n_iter, n_bins=n_bins).round(3)

    # Store shape property features
    shape_features = np.concatenate((a3, d1, d2, d3, d4))

    # Store all features
    return np.concatenate((global_features, shape_features))


def extract_features(fp_data: str,  fp_csv_out: str, n_categories: int = 0, n_iter: int = 1_000, n_bins: int = 10) -> None:
    categories = next(os.walk(fp_data))[1]
    n_categories = len(categories) if not n_categories else n_categories
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
            full_filename = f"{category}/{filename}"

            # Calculate features of current mesh
            mesh_features = calculate_mesh_features(fp_mesh, full_filename, category, n_iter=n_iter, n_bins=n_bins)
            all_features.append(mesh_features)

    hists = ","
    for feature in ["a3", "d1", "d2", "d3", "d4"]:
        hists += ",".join([f"{feature}_{i}" for i in range(n_bins)]) + ","

    # Save data to CSV
    header = "filename,category,surface_area,volume,compactness,diameter,convexity,eccentricity,rectangularity" + hists[:-1]  # Remove last comma

    # Comments='' removes the '#' character from the header
    np.savetxt(fp_csv_out, all_features, delimiter=",", fmt="%s", header=header, comments="")


if __name__ == "__main__":
    fp_data = "./data"
    fp_csv_out = "./csvs/feature_extraction.csv"
    n_categories = 3  # len(categories)
    n_iter = 1_000
    n_bins = 10

    ''' Global features '''
    # Surface area of mesh
    # Mesh volume
    # Compactness: (surface_area ** 1.5) / (36 * math.pi * (mesh_volume ** 0.5))
    # Diameter
    # Convexity: mesh volume over convex hul volume
    # Eccentricity: ratio of largest to smallest eigenvalues of covariance matrix

    ''' Shape property features '''
    # A3: angle between 3 random vertices
    # D1: distance between barycenter and random vertex
    # D2: distance between 2 random vertices
    # D3: square root of area of triangle given by 3 random vertices
    # D4: cube root of volume of tetrahedron formed by 4 random vertices

    extract_features(fp_data=fp_data, fp_csv_out=fp_csv_out, n_categories=n_categories, n_iter=n_iter, n_bins=n_bins)
