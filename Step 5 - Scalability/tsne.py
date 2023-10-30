from sklearn.manifold import TSNE
import numpy as np
import csv
import open3d as o3d
from matplotlib import pyplot as plt


def get_all_features(features_path, exclude=None):
    total_features = []
    labels = []
    paths = []
    with open(features_path, newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=';', quotechar='|')
        
        for i, row in enumerate(csv_reader):
            if i == 0:
                continue # Skip header row
            filepath = row[0]
            if exclude and filepath == exclude:
                continue # Exclude shape from returned features list if specified
            label = row.pop(1)
            row.pop(0)
            features = [float(feature.replace(" ", "")) for feature in row if feature != " "]
            total_features.append(features)
            labels.append(label)
            paths.append(filepath)
    
    return np.array(total_features), labels, paths


def main():
    # Parameters
    features_path = "miniDB/features.csv"
    tsne_no_components = 2
    tsne_perplexity = 3

    # Load feature vectors for db shapes
    features_total, labels, paths = get_all_features(features_path)

    # Embed feature vectors in lower-dimensional space using T-distributed Stochastic Neighbor Embedding (t-SNE)
    features_embedded = TSNE(n_components=tsne_no_components, perplexity=tsne_perplexity).fit_transform(features_total)

    # --- Visualize results --- #
    # Create colors map
    colors_map = {}
    prev_color = 0
    for label in labels:
        if label not in colors_map.keys():
            colors_map[label] = prev_color + 0.1
            prev_color += 0.3

    # Create colors per class
    colors = []
    for label in labels:
        colors.append(colors_map[label])
    colors = np.array(colors)

    # Create plot
    fig, ax = plt.subplots()
    ax.scatter(features_embedded[:,0], features_embedded[:,1], c=colors)
    for path, x, y in zip(paths, features_embedded[:,0], features_embedded[:,1]):
        ax.annotate(path.replace("/m", "-").replace(".obj", ""), (x, y))
    plt.show()


if __name__ == "__main__":
    main()
