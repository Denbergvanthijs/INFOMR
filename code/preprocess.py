import os
import numpy as np
import csv
import pymeshlab

save_mesh_info = 0

def read_meshes():
    ms = pymeshlab.MeshSet()

    data_folder = 'data'

    # Subset of the data
    # categories = ['DeskLamp', 'Bottle', 'Skyscraper']
    categories = next(os.walk(data_folder))[1]

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
                ms.load_new_mesh(mesh_path)
                m = ms.current_mesh()

                vertices = m.vertex_number()
                faces = m.face_number()
                bbox = m.bounding_box()
                bbox_min = bbox.min()
                bbox_max = bbox.max()

                # Now we only store class label, num vertices and num faces. We should be more complete of course
                mesh_info.append([category, vertices, faces, 
                                  bbox_min[0], bbox_min[1], bbox_min[2], 
                                  bbox_max[0], bbox_max[1], bbox_max[2]])

                # vertices = []
                # faces = []

                # with open(mesh_path, 'r') as file:
                #     lines = file.readlines()

                # for line in lines:
                #     parts = line.strip().split()

                #     if not parts:
                #         continue

                #     if parts[0] == 'v':
                #         vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
                #         vertices.append(vertex)
                #     elif parts[0] == 'f':
                #         face = [int(v.split('/')[0]) - 1 for v in parts[1:]]  
                #         faces.append(face)

                # # Now we only store class label, num vertices and num faces. We should be more complete ofcourse
                # mesh_info.append([category, len(vertices), len(faces)])

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
    mesh_info = read_meshes()

    if save_mesh_info:
        # Save mesh info to a CSV file
        csv_file_path = "data/mesh_info.csv"
        with open(csv_file_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Class', 'Vertices', 'Faces', 
                                'BB Min x', 'BB Min y', 'BB Min z', 
                                'BB Max x', 'BB Max y', 'BB Max z'])
            for row in mesh_info:
                csv_writer.writerow(row)

    print(avg_shape(mesh_info))
