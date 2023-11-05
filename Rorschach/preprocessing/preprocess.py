import csv
import os

import numpy as np
import pandas as pd
import pymeshlab
from tqdm import tqdm


def sign_error(a, b=[-1.0, -1.0, -1.0]):
    if np.shape(a) != np.shape(b):
        raise RuntimeError("Oops! The arrays have different shapes!")

    return np.sum(np.array(a) - np.array(b))


def calc_vertices_sign(vertices: np.ndarray) -> tuple:
    if np.shape(vertices)[1] != 3:
        raise RuntimeError("Oops! The vertices array has the wrong shape!")

    signx, signy, signz = np.sign(np.sum(np.sign(vertices) * np.square(vertices), axis=0))

    return signx, signy, signz


def flip_flags(signx: float, signy: float, signz: float) -> tuple:
    flip_x = signx > 0  # Flip along YZ plane
    flip_y = signy > 0  # Flip along XZ plane
    flip_z = signz > 0  # Flip along XY plane

    return flip_x, flip_y, flip_z


def rows_sqr_error(m1, m2=np.identity(3)):
    if np.shape(m1) != np.shape(m2):
        raise RuntimeError("Oops! The matrices have different shapes!")

    errors = np.sum(np.square(m1), axis=1) - np.sum(np.square(m2), axis=1)

    return np.sum(errors)


def normalize_mesh(meshset: pymeshlab.MeshSet) -> pymeshlab.MeshSet:
    measures = meshset.get_geometric_measures()  # Dictionary with geometric measures
    # All keys of measures dict:
    # "barycenter"
    # "shell_barycenter"
    # "pca"
    # "bbox"
    # "surface_area"
    # "total_edge_inc_faux_length"
    # "total_edge_length"
    # "avg_edge_inc_faux_length"
    # "avg_edge_length"
    barycenter = measures["barycenter"]  # Compute barycenter before translation
    # print(f"bc: {np.sum(np.square(measures['barycenter']))}")

    ### Translate: baricenter to origin ###
    meshset.apply_filter("compute_matrix_from_translation",
                         traslmethod=3,  # traslmethod of 3 is to set a new origin instead of translating
                         neworigin=measures["barycenter"])  # Use point clod for barycenter and make that new origin

    measures = meshset.get_geometric_measures()  # Compute measures again with new barycenter
    barycenter = measures["barycenter"]
    pca_axes = measures["pca"]
    # print(f"bc after trans: {np.sum(np.square(barycenter))}")
    # print(f"pc:\n{pca_axes.round(3)}")

    ### Pose: Rotate to align axes ###
    meshset.apply_filter("compute_matrix_by_principal_axis", pointsflag=True, freeze=True, alllayers=True)

    measures = meshset.get_geometric_measures()  # Compute measures again after axis align
    pca_axes = measures["pca"]
    # print(f"Aligned pc:\n{pca_axes.round(3)}")
    # print(f"bc after rot: {np.sum(np.square(measures['barycenter']))}")

    ### Flip: heavy side to negative ###
    mesh = meshset.current_mesh()
    vertices = mesh.vertex_matrix()
    signx, signy, signz = calc_vertices_sign(vertices)  # Compute second order moment (SOM) sign

    # print(f"SOM sign: [{signx}  {signy}  {signz}]")

    # Flip axes whose pc is in the positive direction (to flip more mass towards negative side)
    flip_x, flip_y, flip_z = flip_flags(signx, signy, signz)  # Whether to flip along any of the axes

    meshset.apply_filter("apply_matrix_flip_or_swap_axis", flipx=flip_x, flipy=flip_y, flipz=flip_z)

    mesh = meshset.current_mesh()
    vertices = mesh.vertex_matrix()
    signx, signy, signz = calc_vertices_sign(vertices)  # Compute measures again after flipping

    # print(f"SOM flipped: [{signx}  {signy}  {signz}]")
    bbox = measures["bbox"]
    bbox_max_dim = max(bbox.dim_x(), bbox.dim_y(), bbox.dim_z())
    # print(f"Max dim: {bbox_max_dim}")
    # print(f"bc after flip: {np.sum(np.square(measures['barycenter']))}")

    ### Size: multiply every dimension by inverse biggest axis ###
    meshset.apply_filter("compute_matrix_from_scaling_or_normalization",
                         scalecenter=0,  # scalecenter=0 to scale around world origin, 1 to scale around barycenter
                         unitflag=True)  # unitflag=True to scale to unit cube

    measures = meshset.get_geometric_measures()  # Compute measures again after scaling
    bbox = measures["bbox"]
    bbox_max_dim = max(bbox.dim_x(), bbox.dim_y(), bbox.dim_z())
    # print(f"Scaled max dim: {bbox_max_dim}")
    # print(f"bc after scaling: {np.sum(np.square(measures['barycenter']))}")

    return meshset


def read_meshes(data_folder: str = "data_cleaned", data_folder_output: str = "data_normalized", n_categories: int = 0) -> np.ndarray:
    """Reads meshes from data folder and returns a numpy array with mesh info.

    :param data_folder: Path to ShapeDatabase folder, defaults to "data"
    :type data_folder: str, optional
    :param n_categories: Decrease number of categories for debugging purposes, defaults to False
    :type n_categories: int, optional
    :return: Numpy array with mesh info
    :rtype: np.ndarray
    """
    meshset = pymeshlab.MeshSet()
    meshset_normalized = pymeshlab.MeshSet()

    categories = next(os.walk(data_folder))[1]
    n_categories = len(categories) if not n_categories else n_categories
    print(f"Reading {n_categories} categories from {data_folder}...")

    # Make output folder if it does not exist
    if not os.path.exists(data_folder_output):
        os.makedirs(data_folder_output)

    # Initial list to store all mesh info
    mesh_info = []
    mesh_info_normalized = []

    # Iterate over all classes in the dataset (desklamp, bottle etc.)
    for category in tqdm(categories[:n_categories]):
        fp_cat_in = os.path.join(data_folder, category)  # Input folder
        fp_cat_out = os.path.join(data_folder_output, category)  # Output folder

        if not os.path.exists(fp_cat_in):
            print(f"\nThe '{category}' folder does not exist.")
            continue

        # Make subfolder for current category in output folder
        if not os.path.exists(fp_cat_out):
            os.makedirs(fp_cat_out)

        # Iterate over all mesh files in current subfolder
        for filename in os.listdir(fp_cat_in):
            fp_mesh = os.path.join(fp_cat_in, filename)  # Input mesh file
            fp_mesh_out = os.path.join(fp_cat_out, filename)  # Output mesh file

            # Load a single mesh
            if not os.path.isfile(fp_mesh):
                print(f"\nThe '{filename}' file does not exist.")
                continue

            meshset_normalized.load_new_mesh(fp_mesh)
            # TODO: fix meshes with inconsistent normals and holes (see technical tip 3b) before normalizing
            meshset_normalized = normalize_mesh(meshset_normalized)
            mesh_normalized = meshset_normalized.current_mesh()

            # Write normalized mesh to output folder
            meshset_normalized.save_current_mesh(fp_mesh_out)

            meshset.load_new_mesh(fp_mesh)
            mesh = meshset.current_mesh()
            measures = meshset.get_geometric_measures()  # Dictionary with geometric measures
            measures_normalized = meshset_normalized.get_geometric_measures()

            # Obtain mesh information

            # Number of vertices
            v_no = mesh.vertex_number()
            v_no_normalized = mesh_normalized.vertex_number()

            # Number of faces
            f_no = mesh.face_number()
            f_no_normalized = mesh_normalized.face_number()

            # Mesh barycenter squared distance to origin
            # Expected value after translation: 0
            bary_sqr_dist = np.sum(np.square(measures["barycenter"]))
            bary_sqr_dist_normalized = np.sum(np.square(measures_normalized["barycenter"]))

            # Principal component "squared error"
            # Add up the squares of each coordinate for each principal component and subtract from identity matrix
            # reason: after alignment, principal components should be 1 in each global axis dir
            # Expected value after alignment: 0
            pc_sqr_error = rows_sqr_error(measures["pca"])
            pc_sqr_error_normalized = rows_sqr_error(measures_normalized["pca"])

            # Second order moment "error"
            # Count of how many axes we have to flip to orient "massive" parts in negative direction (global axis)
            # Expected value after flip: 0
            vertices = mesh.vertex_matrix()
            signx, signy, signz = calc_vertices_sign(vertices)
            som_error = sign_error([signx, signy, signz])

            vertices_normalized = mesh_normalized.vertex_matrix()
            signx2, signy2, signz2 = calc_vertices_sign(vertices_normalized)
            som_error_normalized = sign_error([signx2, signy2, signz2])

            # Biggest dimension of the bounding box containing the mesh
            # Expected value after resize: 1
            bbox = measures["bbox"]
            bbox_normalized = measures_normalized["bbox"]

            max_dim = max(bbox.dim_x(), bbox.dim_y(), bbox.dim_z())
            max_dim_normalized = max(bbox_normalized.dim_x(),
                                     bbox_normalized.dim_y(),
                                     bbox_normalized.dim_z())

            # Print old and normalized vertex and face count
            # print(f"Old: {vertices} vertices, {faces} faces; normalized: {vertices_normalized} vertices, {faces_normalized} faces")

            # Print old and normalized bounding box
            # print(f"Old: {bbox_min.round(3)} - {bbox_max.round(3)}; normalized: {bbox_min_normalized.round(3)} - {bbox_max_normalized.round(3)}\n")

            mesh_info.append([filename,
                              category,
                              v_no,
                              f_no,
                              bary_sqr_dist,
                              pc_sqr_error,
                              som_error,
                              max_dim])

            mesh_info_normalized.append([filename,
                                         category,
                                         v_no_normalized,
                                         f_no_normalized,
                                         bary_sqr_dist_normalized,
                                         pc_sqr_error_normalized,
                                         som_error_normalized,
                                         max_dim_normalized])

        meshset.clear()
        meshset_normalized.clear()

    return [np.array(mesh_info), np.array(mesh_info_normalized)]


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
    # Flags to read/write info from csv
    read_mesh_info = False
    save_mesh_info = True
    csv_file_path = "./data_cleaned/mesh_info.csv"
    csv_fp_normalized = "./data_normalized/mesh_info.csv"

    fp_meshes_in = "./data_cleaned"  # Input, the refined meshes, i.e. after running patch_meshes.py
    fp_meshes_out = "./data_normalized"  # Output, the normalised meshes
    n_categories = 0  # 0 to read all categories

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
        # print(f"Average shape: {n_vertices:_.0f} vertices, {n_faces:_.0f} faces")

    # Read mesh info from data folder and save it to a CSV file
    elif save_mesh_info:
        column_names = ["Filename", "Class", "Vertices", "Faces",
                        "Barycenter offset", "Principal comp error", "SOM error", "Max dim"]
        mesh_info, mesh_info_normalized = read_meshes(data_folder=fp_meshes_in,
                                                      data_folder_output=fp_meshes_out,
                                                      n_categories=n_categories)

        # Save mesh info in a CSV file
        with open(csv_file_path, "w", newline="", encoding="utf-8") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(column_names)

            for row in mesh_info:
                csv_writer.writerow(row)

        # Save normalized mesh info in a CSV file
        with open(csv_fp_normalized, "w", newline="", encoding="utf-8") as csvnorm:
            csv_writer = csv.writer(csvnorm)
            csv_writer.writerow(column_names)

            for row in mesh_info_normalized:
                csv_writer.writerow(row)

        # Print average shape
        n_vertices, n_faces = avg_shape(mesh_info)  # 5609 vertices, 10691 faces
        # print(f"Average shape: {n_vertices:_.0f} vertices, {n_faces:_.0f} faces")

    # Read mesh info from data folder but don't save
    else:
        mesh_info, mesh_info_normalized = read_meshes()

        # Print average shape
        n_vertices, n_faces = avg_shape(mesh_info)  # 5609 vertices, 10691 faces
        # print(f"Average shape: {n_vertices:_.0f} vertices, {n_faces:_.0f} faces")

        # Full preprocessing (normalization) takes 6 minutes 17 seconds (Riemer)
