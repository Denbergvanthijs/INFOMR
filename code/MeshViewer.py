import argparse

import open3d as o3d
import pymeshlab

parser = argparse.ArgumentParser(description="Visualize a given mesh.")
parser.add_argument("--mesh_path", type=str, dest="mesh_path",
                    default="../data/Spoon/D00014.obj", help="Path to the mesh file.")
parser.add_argument("--visualization_method", type=str, dest="visualization_method",
                    default="shade", help="Visualization method. Either 'shade' or 'wired'.", choices=["shade", "wired"])


def show_info(mesh_path: str) -> (int, int):
    """Function for retrieving number of vertices and faces of a given mesh.

    :param mesh_path: Path to the mesh file
    :type mesh_path: str

    :return: Number of vertices and faces of the given mesh
    :rtype: (int, int)
    """
    # Obtain mesh information
    meshset = pymeshlab.MeshSet()
    meshset.load_new_mesh(mesh_path)
    mesh_current = meshset.current_mesh()

    return mesh_current.vertex_number(), mesh_current.face_number()


# Function for visualizing a given mesh
def visualize(mesh_path: str, visualization_method: str, width: int = 1280, height: int = 720) -> None:
    """Function for visualizing a given mesh.

    :param mesh_path: Path to the mesh file
    :type mesh_path: str
    :param visualization_method: Visualization method. Either 'shade' or 'wired'.
    :type visualization_method: str

    :raises IndexError: If no acceptable visualization method was given

    :return: None
    :rtype: None
    """
    # Load the mesh with open3d
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()

    window_name = f"RorschachViz - Visualized mesh: {mesh_path}"

    # Visualize the mesh
    if visualization_method == "shade":
        o3d.visualization.draw_geometries([mesh], width=width, height=height, window_name=window_name)
    elif visualization_method == "wired":
        o3d.visualization.draw_geometries([mesh], width=width, height=height, window_name=window_name, mesh_show_wireframe=True)
    else:
        raise IndexError("No acceptable visualization method was given. Please enter either 'shade' or 'wired' as an argument.")


if __name__ == "__main__":
    # Obtain filename from command line input
    # Example command: python visualize_mesh.py --mesh_path ./data/LabeledDB_new/Ant/81.off --visualization_method shade
    args = parser.parse_args()

    vertices, faces = show_info(args.mesh_path)
    print(f"Number of vertices: {vertices}")
    print(f"Number of faces: {faces}")

    visualize(args.mesh_path, args.visualization_method)
