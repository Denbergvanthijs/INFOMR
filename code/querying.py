
import csv

import open3d as o3d
from distance_functions import get_emd


def get_features(features_path, mesh_path):
    with open(features_path, newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for i, row in enumerate(csv_reader):
            if i == 0:
                continue
            filepath = row[0]
            if filepath == mesh_path:
                label = row.pop(1)
                row.pop(0)
                features = [float(feature.replace(" ", "")) for feature in row if feature != " "]
                return features, label
        else:
            raise RuntimeError(f"Mesh path {mesh_path} not found in database.")


def load_meshes(meshpaths):
    # Load meshes
    meshes = []
    for i, meshpath in enumerate(meshpaths):
        mesh = o3d.io.read_triangle_mesh("data/" + meshpath)
        mesh.compute_vertex_normals()

        # Add translation offset
        mesh.translate((i, 0, 0))
        meshes.append(mesh)

    return meshes


def visualize(meshes):
    o3d.visualization.draw_geometries(
        meshes,
        width=1280,
        height=720,
        mesh_show_wireframe=True
    )


def main():
    # Parameters
    query_path = "Humanoid/m236.obj"
    mesh1_path = "Knife/D01119.obj"
    mesh2_path = "Humanoid/m158.obj"
    mesh3_path = "Bicycle/D00077.obj"
    features_path = "csvs/feature_extraction.csv"

    meshes = load_meshes([query_path, mesh1_path, mesh2_path, mesh3_path])
    features_query, label1 = get_features(features_path, query_path)
    features_1, label2 = get_features(features_path, mesh1_path)
    features_2, label3 = get_features(features_path, mesh2_path)
    features_3, label4 = get_features(features_path, mesh3_path)

    print(f"Earth Mover's Distance between {query_path} and {mesh1_path}: {get_emd(features_query, features_1)}")
    print(f"Earth Mover's Distance between {query_path} and {mesh2_path}: {get_emd(features_query, features_2)}")
    print(f"Earth Mover's Distance between {query_path} and {mesh3_path}: {get_emd(features_query, features_3)}")

    visualize(meshes)


if __name__ == "__main__":
    main()
