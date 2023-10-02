import csv
import os

import numpy as np
import pandas as pd
import pymeshfix
import pymeshlab
from tqdm import tqdm

if __name__ == "__main__":
    meshset = pymeshlab.MeshSet()
    meshset_clean = pymeshlab.MeshSet()

    categories = next(os.walk("data"))[1]
    n_categories = 1  # len(categories)
    print(f"Reading {n_categories} categories from /data...")

    # Iterate over all classes in the dataset (desklamp, bottle etc.)
    for category in tqdm(categories[:n_categories]):
        fp_cat_in = os.path.join("data", category)  # Input folder
        fp_cat_out = os.path.join("data_cleaned", category)  # Output folder

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

            meshset.load_new_mesh(fp_mesh)
            mesh = meshset.current_mesh()
            vertices = mesh.vertex_matrix()
            faces = mesh.face_matrix()

            vclean, fclean = pymeshfix.clean_from_arrays(vertices, faces)

            mesh = pymeshlab.Mesh(vclean, fclean)

            meshset.add_mesh(mesh)

            meshset.save_current_mesh(fp_mesh_out)
