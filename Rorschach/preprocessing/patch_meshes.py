import csv
import os

import numpy as np
import open3d as o3d
import pymeshfix
from pymeshlab import AbsoluteValue, Mesh, MeshSet
from tqdm import tqdm


def reorient_normals(fp_data: str, fp_data_out: str = "data_cleaned", n_categories: int = 0):
    categories = next(os.walk(fp_data))[1]
    print(f"Reading {n_categories} categories from {fp_data}...")

    # Iterate over all classes in the dataset (desklamp, bottle etc.)
    for category in tqdm(categories[:n_categories]):
        fp_cat_in = os.path.join(fp_data, category)  # Input folder
        fp_cat_out = os.path.join(fp_data_out, category)  # Output folder

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
            fp_mesh_out = fp_mesh_out.replace(".obj", ".ply")

            # Load mesh
            mesh = o3d.io.read_triangle_mesh(fp_mesh)
            mesh.compute_vertex_normals()

            # Generate point cloud from triangle mesh
            # pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(np.asarray(mesh.vertices)))
            pcd = mesh.sample_points_poisson_disk(5000)
            # pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))  # invalidate existing normals
            pcd.estimate_normals()
            # o3d.visualization.draw_geometries([pcd], point_show_normal=True)
            pcd.orient_normals_consistent_tangent_plane(100)  # reorient normals with knn
            # o3d.visualization.draw_geometries([pcd], point_show_normal=True)

            reconstructed_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=10)
            reconstructed_mesh.paint_uniform_color(np.array([[0.5], [0.5], [0.5]]))

            # mesh = mesh.remove_duplicated_vertices()
            # mesh = mesh.remove_degenerate_triangles()
            # mesh = mesh.remove_duplicated_triangles()
            # mesh = mesh.remove_non_manifold_edges()
            mesh.orient_triangles()

            # Save mesh
            o3d.io.write_triangle_mesh(fp_mesh_out, reconstructed_mesh)


def clean_mesh(fp_mesh: str, fp_mesh_out: str, cleanMeshes: bool = True, remeshTargVert: int = 0) -> tuple:
    meshset = MeshSet()
    meshset.load_new_mesh(fp_mesh)  # Load mesh

    mesh = meshset.current_mesh()
    v_no = mesh.vertex_number()  # Number of vertices before cleaning
    f_no = mesh.face_number()  # Number of faces before cleaning

    # Clean mesh
    if cleanMeshes:
        vertices = mesh.vertex_matrix()
        faces = mesh.face_matrix()
        vclean, fclean = pymeshfix.clean_from_arrays(vertices, faces)

        # Check for empty mesh
        if vclean.size == 0 or fclean.size == 0:
            # Keep original mesh in dataset and do not overwrite the mesh variable
            errors.append(f"{fp_mesh} has {vclean.size} vertices and {fclean.size} faces.")
        else:
            # Create new mesh from cleaned vertices and faces
            mesh = Mesh(vclean, fclean)
            meshset.add_mesh(mesh)

    # Remeshing
    if remeshTargVert > 0:
        if v_no < remeshTargVert * 0.5:
            meshset.meshing_isotropic_explicit_remeshing(targetlen=AbsoluteValue(0.02), iterations=4)

    # Save cleaned mesh
    meshset.save_current_mesh(fp_mesh_out)

    # Return number of vertices and faces before cleaning
    return v_no, f_no


def refine_meshes(fp_data: str = "data", fp_data_out: str = "data_cleaned", n_categories: int = 0,
                  remeshTargVert=0, cleanMeshes=True, saveRawInfo=False) -> list:
    categories = next(os.walk(fp_data))[1]
    print(f"Reading {n_categories} categories from {fp_data}...")

    errors = []
    mesh_info = []

    # Iterate over all classes in the dataset (desklamp, bottle etc.)
    for category in tqdm(categories[:n_categories]):
        fp_cat_in = os.path.join(fp_data, category)  # Input folder
        fp_cat_out = os.path.join(fp_data_out, category)  # Output folder

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

            # Number of vertices and faces before cleaning
            v_no, f_no = clean_mesh(fp_mesh, fp_mesh_out, cleanMeshes=cleanMeshes, remeshTargVert=remeshTargVert)

            mesh_info.append([filename, category, v_no, f_no])

    if saveRawInfo:
        # Save mesh info in a CSV file
        csv_file_path = os.path.join(fp_data, "mesh_info.csv")
        column_names = ["Filename", "Class", "Vertices", "Faces"]
        with open(csv_file_path, "w", newline="", encoding="utf-8") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(column_names)

            mesh_info = np.array(mesh_info)

            for row in mesh_info:
                csv_writer.writerow(row)

    return errors


def fill_holes(fp_data: str = "data", fp_data_out: str = "data_cleaned", n_categories: int = -1):
    categories = next(os.walk(fp_data))[1]
    print(f"Reading {n_categories} categories from {fp_data}...")

    # Iterate over all classes in the dataset (desklamp, bottle etc.)
    for category in tqdm(categories[:n_categories]):
        fp_cat_in = os.path.join(fp_data, category)  # Input folder
        fp_cat_out = os.path.join(fp_data_out, category)  # Output folder

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

            # Load mesh
            mesh = o3d.io.read_triangle_mesh(fp_mesh)
            mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh).fill_holes().to_legacy()

            # Save mesh
            o3d.io.write_triangle_mesh(fp_mesh_out, mesh)


def count_nonwatertight(fp_data: str = "data_cleaned"):
    categories = next(os.walk(fp_data))[1]
    print(f"Reading {n_categories} categories from {fp_data}...")

    nonWatertight = 0

    # Iterate over all classes in the dataset (desklamp, bottle etc.)
    for category in tqdm(categories):
        cat_dir = os.path.join(fp_data, category)

        if not os.path.exists(cat_dir):
            print(f"\nThe '{category}' folder does not exist.")
            continue

        # Iterate over all mesh files in current subfolder
        for filename in os.listdir(cat_dir):
            fp_mesh = os.path.join(cat_dir, filename)

            # Load mesh
            mesh = o3d.io.read_triangle_mesh(fp_mesh)
            mesh.compute_vertex_normals()

            if not mesh.is_watertight():
                nonWatertight += 1

    return nonWatertight


if __name__ == "__main__":
    fp_data = "data"
    fp_data_out = "data_cleaned"
    n_categories = 70  # len(categories)

    # print(f"\nThere are {count_nonwatertight(fp_data)} non-watertight meshes in the dataset before processing.")

    # Reorient mesh normals to be consistent
    # open3d existing method is not doing anything for some reason. Tried surface reconstruction with poor results.
    # reorient_normals(fp_data=fp_data, fp_data_out=fp_data_out, n_categories=n_categories)

    # Fill holes
    # Bad register allocation error. I think the legacy version of the triangle meshes used by this method can't handle some of our bigger meshes.
    # fill_holes(fp_data=fp_data, fp_data_out=fp_data_out, n_categories=n_categories)

    # Stitch mesh holes and clean
    # Cleaning method significantly trims some of the meshes. Not worth using.
    # errors = refine_meshes(fp_data=fp_data, fp_data_out=fp_data_out, n_categories=n_categories)
    errors = refine_meshes(fp_data=fp_data, fp_data_out=fp_data_out, n_categories=n_categories,
                           remeshTargVert=10_000, cleanMeshes=False, saveRawInfo=True)
    print(f"{len(errors)} errors found while stitching:\n{errors}")

    # print(f"\nThere are {count_nonwatertight(fp_data_out)} non-watertight meshes in the dataset after processing.")

    # Patching and cleaning meshes of entire dataset takes 1 hour 13 minutes (Riemer)
