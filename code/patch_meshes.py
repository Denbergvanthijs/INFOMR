import os

import pymeshfix
from pymeshlab import Mesh, MeshSet
from tqdm import tqdm


def refine_meshes(fp_data: str, fp_data_out: str = "data_cleaned", n_categories: int = 0) -> list:
    meshset = MeshSet()

    categories = next(os.walk(fp_data))[1]
    print(f"Reading {n_categories} categories from {fp_data}...")

    errors = []

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
            meshset.load_new_mesh(fp_mesh)
            mesh = meshset.current_mesh()

            # Clean mesh
            vertices = mesh.vertex_matrix()
            faces = mesh.face_matrix()
            vclean, fclean = pymeshfix.clean_from_arrays(vertices, faces)

            # Check for empty mesh
            if vclean.size == 0 or fclean.size == 0:
                # Keep original mesh in dataset and do not overwrite the mesh variable
                errors.append(f"{filename} has {vclean.size} vertices and {fclean.size} faces.")
            else:
                # Create new mesh from cleaned vertices and faces
                mesh = Mesh(vclean, fclean)

            # Save cleaned mesh
            meshset.add_mesh(mesh)
            meshset.save_current_mesh(fp_mesh_out)

    return errors


if __name__ == "__main__":
    fp_data = "data"
    fp_data_out = "data_cleaned"
    n_categories = 69  # len(categories)

    errors = refine_meshes(fp_data=fp_data, fp_data_out=fp_data_out, n_categories=n_categories)
    print(errors)

    # Patching and cleaning meshes of entire dataset takes 1 hour 13 minutes (Riemer)
