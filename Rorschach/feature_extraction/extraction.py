import itertools
import math
import os

import numpy as np
import open3d as o3d
from pymeshlab import MeshSet
from tqdm import tqdm
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay


def compute_area(mesh) -> tuple:
    """Compute the area and volume of a mesh

    :param mesh: Mesh to compute area and volume of
    :type mesh: open3d
    :return: Area and volume of mesh
    :rtype: tuple
    """
    # Compute mesh area and volume
    area = mesh.get_surface_area()

    # if mesh.is_watertight():  # Only compute volume if mesh is watertight
    #     volume = mesh.get_volume()
    # else:
    #     volume = -1

    return area


# Compute volume of convex hull of a mesh
def compute_convex_hull_volume(vertices):
    convex_hull = ConvexHull(vertices)
    hull_volume = convex_hull.volume
    return hull_volume


# def tetrahedron_volume(a, b, c, d):
#     return np.abs(np.einsum('ij,ij->i', a-d, np.cross(b-d, c-d))) / 6


# def compute_convex_hull_volume(pts):
#     ch = ConvexHull(pts)
#     dt = Delaunay(pts[ch.vertices])
#     tets = dt.points[dt.simplices]
#     return np.sum(tetrahedron_volume(tets[:, 0], tets[:, 1],
#                                      tets[:, 2], tets[:, 3]))


# Compute the Oriented Bounding Box (obb) and return its volume
def compute_obb_volume(mesh):
    # aabb = mesh.get_axis_aligned_bounding_box()
    # aabb_volume = aabb.volume()
    obb = mesh.get_oriented_bounding_box()
    obb_volume = obb.volume()
    return obb_volume


# Compute compactness of a mesh (based on mesh area and volume)
def compute_compactness(area, volume):
    # Return -1 if volume is invalid due to previous error
    # if volume == -1:
    #     return -1

    compactness = (area ** 1.5) / (36 * math.pi * (volume ** 0.5))
    return compactness


# Compute the diameter of a mesh
def compute_diameter(vertices, barycenter):
    # Calculate distance from center to all vertices
    distances = np.linalg.norm(vertices - barycenter, axis=1)

    # Calculate diameter as twice the distance of center to furthest vertice
    diameter = 2 * np.max(distances)
    return diameter


# Compute convexity of a mesh (shape volume over convex hull volume)
def compute_convexity(vertices, shape_volume):
    # Return -1 if there are not enough vertices or not minimal 4 unique x coordinates
    if vertices.shape[0] < 4 or len(np.unique(vertices[:, 0])) < 4:
        return -1

    # Calculate convex hull and corresponding volume
    convex_hull = ConvexHull(vertices)
    convex_hull_volume = convex_hull.volume

    # Divide mesh volume by convex hull volume
    convexity = shape_volume / convex_hull_volume
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
def compute_rectangularity(shape_volume, obb_volume):
    rectangularity = shape_volume / obb_volume
    return rectangularity


def compute_angle_3D(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray) -> float:
    # Based on: https://stackoverflow.com/a/35178910/10603874
    v2v1 = v1 - v2  # Normalized vectors
    v2v3 = v3 - v2

    denominator = np.linalg.norm(v2v1) * np.linalg.norm(v2v3)
    if denominator == 0:  # Avoid division by zero
        cosine_angle = 0
    else:
        cosine_angle = np.dot(v2v1, v2v3) / denominator  # Cosine of angle between vectors

    if -1 <= cosine_angle <= 1:  # Avoid math domain error
        angle = np.arccos(cosine_angle)  # Angle in radians
    else:
        angle = 0

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
    # Load mesh with pymeshlab
    meshset = MeshSet()
    meshset.load_new_mesh(fp_mesh)
    mesh_py = meshset.current_mesh()

    # Load mesh with open3d
    mesh_o3d = o3d.io.read_triangle_mesh(fp_mesh)
    mesh_o3d.compute_vertex_normals()

    # Get data
    vertices = mesh_py.vertex_matrix()
    measures = meshset.get_geometric_measures()  # Dictionary with geometric measures
    barycenter = measures["barycenter"]

    # Compute global features
    area = compute_area(mesh_o3d)
    hull_volume = compute_convex_hull_volume(vertices)
    obb_volume = compute_obb_volume(mesh_o3d)
    compactness = compute_compactness(area, hull_volume)
    diameter = compute_diameter(vertices, barycenter)
    # convexity = compute_convexity(vertices, hull_volume)
    eccentricity = compute_eccentricity(vertices)
    rectangularity = compute_rectangularity(hull_volume, obb_volume)

    # Store global features as well as filename and category
    global_features = np.array([full_filename, category, area, hull_volume, obb_volume, compactness, diameter, eccentricity, rectangularity])

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
    fails = []

    categories = next(os.walk(fp_data))[1]
    print(categories)
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
            try:
                mesh_features = calculate_mesh_features(fp_mesh, full_filename, category, n_iter=n_iter, n_bins=n_bins)
                all_features.append(mesh_features)
            except:
                print('fail')
                print(fp_mesh)
                fails.append(fp_mesh)
                continue

    hists = ","
    for feature in ["a3", "d1", "d2", "d3", "d4"]:
        hists += ",".join([f"{feature}_{i}" for i in range(n_bins)]) + ","

    # Save data to CSV
    header = "filename,category,"
    header += "surface_area,hull_volume,obb_volume,compactness,diameter,eccentricity,rectangularity"
    header += hists[:-1]  # Remove last comma

    print(fails)

    # Comments='' removes the '#' character from the header
    np.savetxt(fp_csv_out, all_features, delimiter=",", fmt="%s", header=header, comments="")

    # Fill NA values
    # df = pd.read_csv('./Rorschach/feature_extraction/features.csv')
    # df = df.fillna(0)

    # # Save the updated DataFrame back to a CSV file
    # df.to_csv('./Rorschach/feature_extraction/features.csv', index=False)


if __name__ == "__main__":
    fp_data = "./data_normalized"
    fp_csv_out = "./Rorschach/feature_extraction/features_new.csv"
    n_categories = 2  # len(categories)
    n_iter = 1_000
    n_bins = 10

    # Set numpy random seed for reproducibility
    # Otherwise A1 and D1 to D4 will change
    np.random.seed(42)

    ''' Global features '''
    # Surface area of mesh
    # Mesh volume
    # Compactness: (surface_area ** 1.5) / (36 * math.pi * (mesh_volume ** 0.5))
    # Diameter
    # Convexity: mesh volume over convex hul volume
    # Eccentricity: ratio of largest to smallest eigenvalues of covariance matrix
    # Rectangularity

    ''' Shape property features '''
    # A3: angle between 3 random vertices
    # D1: distance between barycenter and random vertex
    # D2: distance between 2 random vertices
    # D3: square root of area of triangle given by 3 random vertices
    # D4: cube root of volume of tetrahedron formed by 4 random vertices

    extract_features(fp_data=fp_data, fp_csv_out=fp_csv_out, n_categories=n_categories, n_iter=n_iter, n_bins=n_bins)

    # Complete feature extraction takes 30 minutes 45 seconds (Riemer), 29 minutes 20 seconds after adjustments (diameter etc.)
