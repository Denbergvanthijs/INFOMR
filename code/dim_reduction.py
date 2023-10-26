from sklearn.manifold import TSNE
import numpy as np
import csv
import open3d as o3d
import pandas as pd
from matplotlib import pyplot as plt


def get_features(features_path):
    mesh_paths = []
    categories = []
    features = []

    with open(features_path, newline='') as file:
        csv_reader = csv.reader(file)

        # Skip the first row (header)
        next(csv_reader)

        for row in csv_reader:
            if len(row) >= 1:
                # First element is the mesh path
                mesh_paths.append(row[0])
                # Second element is the category label (Humanoid, Vase, etc.)
                categories.append(row[1])
                # The remainder of the row are the features (excluding 'volume' and 'compactness' for now)
                features.append(row[2:3] + row[5:])

    return mesh_paths, categories, np.array(features)


def main():
    # Parameters
    features_path = "csvs/feature_extraction.csv"
    tsne_no_components = 2
    # Perplexity value / Sigma value (should be between 30-50 according to Alex)
    # It accounts for the number of nearest neighbours that needs to be preserved after dim. reduction
    tsne_perplexity = 10

    # Load feature vectors for db shapes
    mesh_paths, categories, features = get_features(features_path)

    # Embed feature vectors in lower-dimensional space using T-distributed Stochastic Neighbor Embedding (t-SNE)
    features_embedded = TSNE(n_components=tsne_no_components, perplexity=tsne_perplexity, random_state=42).fit_transform(features)

    # Select only a couple of categories for visualization purposes
    i = 7
    categories = categories[:i] + categories[500:500+i] + categories[1000:1000+i] + categories[1500:1500+i]
    features_embedded = np.vstack((features_embedded[:i], features_embedded[500:500+i], features_embedded[1000:1000+i], features_embedded[1500:1500+i]))

    # Create colors map
    colors_map = {}
    prev_color = 0
    for label in categories:
        if label not in colors_map.keys():
            colors_map[label] = prev_color + 0.1
            prev_color += 0.3

    # Create colors per class
    colors = []
    for label in categories:
        colors.append(colors_map[label])
    colors = np.array(colors)

    # Plot category colors and labels
    fig, ax = plt.subplots()
    ax.scatter(features_embedded[:,0], features_embedded[:,1], c=colors)
    for category, x, y in zip(categories, features_embedded[:,0], features_embedded[:,1]):
        ax.annotate(category, (x, y))

    plt.savefig('2D_meshes.png')


if __name__ == "__main__":
    main()