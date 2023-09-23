import csv
import os

import numpy as np
import pandas as pd
import pymeshlab
from tqdm import tqdm


def read_meshes(data_folder: str = "data", n_categories: int = False) -> np.ndarray:
    """Reads meshes from data folder and returns a numpy array with mesh info.

    :param data_folder: Path to ShapeDatabase folder, defaults to "data"
    :type data_folder: str, optional
    :param n_categories: Decrease number of categories for debugging purposes, defaults to False
    :type n_categories: int, optional
    :return: Numpy array with mesh info
    :rtype: np.ndarray
    """
    ms = pymeshlab.MeshSet()
    categories = next(os.walk(data_folder))[1]
    n_categories = len(categories) if not n_categories else n_categories
    print(f"Reading {n_categories} categories from {data_folder}...")

    # Initial list to store all mesh info
    mesh_info = []

    # Iterate over all classes in the dataset (desklamp, bottle etc.)
    for category in tqdm(categories[:n_categories]):
        category_folder = os.path.join(data_folder, category)

        if not os.path.exists(category_folder):
            print(f"The '{category}' folder does not exist.")
            continue

        # Iterate over all mesh files in current subfolder
        for filename in os.listdir(category_folder):
            mesh_path = os.path.join(category_folder, filename)

            # Load a single mesh
            if not os.path.isfile(mesh_path):
                print(f"The '{filename}' file does not exist.")
                continue

            ms.load_new_mesh(mesh_path)
            mesh = ms.current_mesh()

            if mesh_path == "data\AircraftBuoyant\m1337.obj":
                ms.apply_filter("generate_resampled_uniform_mesh", mergeclosevert=True)
                # ms.apply_filter("remove_isolated_pieces_wrt_diameter",mincomponentdiag=pymeshlab.Percentage(5))
                ms.save_current_mesh("data\AircraftBuoyant\\resampled_m1337.obj")
                # ms.save_current_mesh(category_folder + "\\resampled_" + filename)

            # Obtain mesh information
            vertices = mesh.vertex_number()
            faces = mesh.face_number()
            bbox = mesh.bounding_box()
            bbox_min = bbox.min()  # [x, y, z]
            bbox_max = bbox.max()  # [x, y, z]

            mesh_info.append([filename, category, vertices, faces,
                              bbox_min[0], bbox_min[1], bbox_min[2],
                              bbox_max[0], bbox_max[1], bbox_max[2]])

    return np.array(mesh_info)


def avg_shape(mesh_info: np.ndarray) -> tuple:
    """Calculates the average shape of all meshes in the dataset.

    :param mesh_info: Numpy array with mesh info
    :type mesh_info: np.ndarray
    :return: Average number of vertices and faces
    :rtype: tuple
    """
    n_meshes = mesh_info.shape[0]

    # Rewrite to use numpy
    tot_vertices = np.sum(mesh_info[:, 2].astype(int))
    tot_faces = np.sum(mesh_info[:, 3].astype(int))

    return tot_vertices / n_meshes, tot_faces / n_meshes


if __name__ == "__main__":
    # Flag to read/write info from csv
    read_mesh_info = False
    save_mesh_info = False
    csv_file_path = "./data/mesh_info.csv"

    # Load mesh info from existing CSV file
    if read_mesh_info:
        mesh_info = pd.read_csv(csv_file_path)

        # Load vertices and faces
        vertices = []
        faces = []
        for _, mesh in mesh_info.iterrows():
            vertices.append(mesh["Vertices"])
            faces.append(mesh["Faces"])

        # Convert to numpy array
        vertices = np.array(vertices)
        faces = np.array(faces)

        # Print average shape
        n_vertices, n_faces = avg_shape(mesh_info.values)  # Convert to numpy array
        print(f"Average shape: {n_vertices:_.0f} vertices, {n_faces:_.0f} faces")

    # Read mesh info from data folder and save it to a CSV file
    elif save_mesh_info:
        mesh_info = read_meshes()

        # Save mesh info in a CSV file

        with open(csv_file_path, "w", newline="", encoding="utf-8") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["Filename", "Class", "Vertices", "Faces",
                                 "BB Min x", "BB Min y", "BB Min z",
                                 "BB Max x", "BB Max y", "BB Max z"])

            for row in mesh_info:
                csv_writer.writerow(row)

        # Print average shape
        n_vertices, n_faces = avg_shape(mesh_info)  # 5609 vertices, 10691 faces
        print(f"Average shape: {n_vertices:_.0f} vertices, {n_faces:_.0f} faces")

    else:
        mesh_info = read_meshes()

        # Print average shape
        n_vertices, n_faces = avg_shape(mesh_info)  # 5609 vertices, 10691 faces
        print(f"Average shape: {n_vertices:_.0f} vertices, {n_faces:_.0f} faces")
