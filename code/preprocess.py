import os
import numpy as np
import pandas as pd
import csv
import pymeshlab

save_mesh_info = 0

def read_meshes():
    ms = pymeshlab.MeshSet()
    data_folder = 'data'
    categories = next(os.walk(data_folder))[1]

    # Initial list to store all mesh info
    mesh_info = []

    # Iterate over all classes in the dataset (desklamp, bottle etc.)
    for category in categories:
        category_folder = os.path.join(data_folder, category)
        
        if not os.path.exists(category_folder):
            print(f"The '{category}' folder does not exist.")
            continue

        # Iterate over all mesh files in current subfolder
        for filename in os.listdir(category_folder):
            mesh_path = os.path.join(category_folder, filename)
            
            # Load a single mesh
            if os.path.isfile(mesh_path):
                ms.load_new_mesh(mesh_path)
                m = ms.current_mesh()

                # Obtain mesh information
                vertices = m.vertex_number()
                faces = m.face_number()
                bbox = m.bounding_box()
                bbox_min = bbox.min()
                bbox_max = bbox.max()

                mesh_info.append([category, vertices, faces, 
                                  bbox_min[0], bbox_min[1], bbox_min[2], 
                                  bbox_max[0], bbox_max[1], bbox_max[2]])
                
    return np.array(mesh_info)

def avg_shape(mesh_info):
    n = len(mesh_info)
    tot_vertices = 0
    tot_faces = 0
    for m in mesh_info:
        tot_vertices += int(m[1])
        tot_faces += int(m[2])
    
    return [tot_vertices / n, tot_faces / n]


if __name__ == "__main__":
    # Read mesh info from data folder and save it to a CSV file
    if save_mesh_info:
        mesh_info = read_meshes()

        # Save mesh info in a CSV file
        csv_file_path = "data/mesh_info.csv"
        with open(csv_file_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Class', 'Vertices', 'Faces', 
                                'BB Min x', 'BB Min y', 'BB Min z', 
                                'BB Max x', 'BB Max y', 'BB Max z'])
            for row in mesh_info:
                csv_writer.writerow(row)

    # Load mesh info from existing CSV file
    mesh_info = pd.read_csv("data/mesh_info.csv")

    # Load vertices and faces
    vertices = []
    faces = []
    for _, m in mesh_info.iterrows():
        vertices.append(m['Vertices'])
        faces.append(m['Faces'])

    # Convert to numpy array 
    vertices = np.array(vertices)
    faces = np.array(faces)

    # Print average shape
    # print(np.mean(vertices))     5609.783326621023 
    # print(np.mean(faces))        10691.12686266613
    
    
        
        
