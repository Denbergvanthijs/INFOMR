import os
import numpy as np
import csv

def read_meshes():
    data_folder = 'data'

    # Subset of the data
    categories = ['DeskLamp', 'Bottle', 'Skyscraper']

    # Initial list to store all mesh info
    mesh_info = []

    # Iterate over all classes in data subset 
    for category in categories:
        category_folder = os.path.join(data_folder, category)
        
        if not os.path.exists(category_folder):
            print(f"The '{category}' folder does not exist.")
            continue

        # Iterate over the files within the current class folder
        for filename in os.listdir(category_folder):
            mesh_path = os.path.join(category_folder, filename)
            
            if os.path.isfile(mesh_path):
                '''
                Commented stuff below is cleaner than the actual code below. 
                Couldn't find a 'faces' attribute or anything.
                Code below works fine, but not cool or optimal. I was lazy.
                '''
                # mesh = o3d.io.read_triangle_mesh(mesh_path)
                # mesh.compute_vertex_normals()
                # num_vertices = len(np.asarray(mesh.vertices))

                vertices = []
                faces = []

                with open(mesh_path, 'r') as file:
                    lines = file.readlines()

                for line in lines:
                    parts = line.strip().split()

                    if not parts:
                        continue

                    if parts[0] == 'v':
                        vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
                        vertices.append(vertex)
                    elif parts[0] == 'f':
                        face = [int(v.split('/')[0]) - 1 for v in parts[1:]]  
                        faces.append(face)

                # Now we only store class label, num vertices and num faces. We should be more complete ofcourse
                mesh_info.append([category, len(vertices), len(faces)])

    return np.array(mesh_info)


if __name__ == "__main__":
    mesh_info = read_meshes()

    # Save mesh info to a CSV file
    csv_file_path = "mesh_info.csv"
    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Class', 'Vertices', 'Faces'])
        for row in mesh_info:
             csv_writer.writerow(row)