import os

def main():
    data_folder = 'data'
    categories = ['DeskLamp', 'Bottle', 'Skyscraper']

    for category in categories:
        category_folder = os.path.join(data_folder, category)
        
        if not os.path.exists(category_folder):
            print(f"The '{category}' folder does not exist.")
            continue

        # Iterate over the files within the current class folder
        for filename in os.listdir(category_folder):
            mesh_path = os.path.join(category_folder, filename)
            
            if os.path.isfile(mesh_path):
                # Load the mesh with open3d
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

                print(len(vertices))
                print(len(faces))
                


if __name__ == "__main__":
    main()