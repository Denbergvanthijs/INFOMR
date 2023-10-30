import open3d as o3d
import os

class_path = "miniDB/Humanoid"
paths = [class_path + "/" + _class for _class in os.listdir(class_path)]

# Load the meshes with open3d
meshes = []
print("filenames: ")
for i, path in enumerate(paths):
    print(os.path.basename(path), end="    ")
    mesh = o3d.io.read_triangle_mesh(path)
    mesh.compute_vertex_normals()
    mesh.translate([i*1, 0, 0])
    meshes.append(mesh)

# Visualize the mesh
o3d.visualization.draw_geometries(meshes, width=1280, height=720)

