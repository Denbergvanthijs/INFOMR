import argparse
import open3d as o3d
from pymeshlab import MeshSet


parser = argparse.ArgumentParser(description="Visualize a given mesh.")
parser.add_argument("--mesh_path", type=str, dest="mesh_path",
                    default="./data/Quadruped/D00409.obj", help="Path to the mesh file.")
parser.add_argument("--method", type=str, dest="method",
                    default="standard", help="Visualization method ('standard', 'convex_hull', 'axes')")


def show_info(mesh_path: str) -> (int, int):
    """
    Function for retrieving number of vertices and faces of a given mesh.

    :param mesh_path: Path to the mesh file
    :type mesh_path: str

    :return: Number of vertices and faces of the given mesh
    :rtype: (int, int)
    """
    # Obtain mesh information
    meshset = MeshSet()
    meshset.load_new_mesh(mesh_path)
    mesh_current = meshset.current_mesh()

    return mesh_current.vertex_number(), mesh_current.face_number()


# Function for constructing the convex hull of a mesh
def compute_convex_hull(mesh_path):
    meshset = MeshSet()
    meshset.load_new_mesh(mesh_path)
    meshset.generate_convex_hull()
    hull_path = 'convex_hull.obj'
    meshset.save_current_mesh(hull_path)
    return hull_path


# Function for visualizing a given mesh
def visualize(mesh_path: str, width: int = 1280, height: int = 720, method: str = 'standard') -> None:
    """
    Function for visualizing a given mesh.

    :param mesh_path: Path to the mesh file
    :type mesh_path: str

    :return: None
    :rtype: None
    """
    # Load mesh with open3d
    mesh = o3d.io.read_triangle_mesh(args.mesh_path)
    mesh.compute_vertex_normals()

    window_name = f"Rorschach - Visualized mesh: {mesh_path}"

    # Draw cartesian frame of reference
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
    mesh_frame.translate([0,0,0])
    
    # Visualize mesh and its convex hull
    if method == 'convex_hull':
        hull_path = compute_convex_hull(mesh_path)
        convex_hull = o3d.io.read_triangle_mesh(hull_path)
        convex_hull.compute_vertex_normals()
        # convex_hull.paint_uniform_color(1, 0, 0, 0.3)
        o3d.visualization.draw_geometries([mesh, convex_hull], width=width, height=height, window_name=window_name)

    # Visualize mesh and 3D axes (cartesian frame)
    elif method == 'axes':
        o3d.visualization.draw_geometries([mesh, mesh_frame], width=width, height=height, window_name=window_name)

    # Only visualize the original mesh
    else:
        o3d.visualization.draw_geometries([mesh], width=width, height=height, window_name=window_name)


if __name__ == "__main__":
    # Obtain filename from command line input
    # Example command: python Rorschach/visualization/MeshViewer.py --mesh_path ./data/Spoon/D00014.obj 
    args = parser.parse_args()
    mesh_path = args.mesh_path
    method = args.method

    # Print mesh info
    vertices, faces = show_info(mesh_path)
    print(f"Number of vertices: {vertices}")
    print(f"Number of faces: {faces}")

    # Visualize the mesh either with or without its convex hull, and with or without 3D axes
    visualize(mesh_path, method=method)