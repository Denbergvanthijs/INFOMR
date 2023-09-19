import open3d as o3d
import sys


def main():
    # Obtain file name from command line input
    mesh_path = sys.argv[1]

    # Load the mesh with open3d
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()

    # Visualize the mesh
    if sys.argv[2] == "shade":
        o3d.visualization.draw_geometries([mesh], width=1280, height=720)
    elif sys.argv[2] == "wired":
        o3d.visualization.draw_geometries([mesh], width=1280, height=720, mesh_show_wireframe=True)
    else:
        print("No acceptable visualization method was given. Please enter either 'shade' or 'wired' as an argument.")
    
   
if __name__ == '__main__':
    main()
