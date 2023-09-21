import open3d as o3d
import sys
import pymeshlab


# Function for retrieving number of vertices and faces of a given mesh
def show_info(mesh_path):
    # Obtain mesh information
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(mesh_path)
    m = ms.current_mesh()

    vertices = m.vertex_number()
    faces = m.face_number()
    print(vertices)
    print(faces)


# Function for visualizing a given mesh
def visualize(mesh_path):
    # Load the mesh with open3d 
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    
    # Visualize the mesh
    try:
        if sys.argv[2] == "shade":
            o3d.visualization.draw_geometries([mesh], width=1280, height=720)
        elif sys.argv[2] == "wired":
            o3d.visualization.draw_geometries([mesh], width=1280, height=720, mesh_show_wireframe=True)
    except IndexError:
        print("No acceptable visualization method was given. Please enter either 'shade' or 'wired' as an argument.")
    
   
if __name__ == "__main__":
    # Obtain filename from command line input
    mesh_path = sys.argv[1]
    show_info(mesh_path)
    visualize(mesh_path)
