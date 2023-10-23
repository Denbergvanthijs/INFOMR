from sklearn.manifold import TSNE
import numpy as np
import csv
import open3d as o3d
import pandas as pd
from matplotlib import pyplot as plt


def get_all_features(features_path, exclude=None):
    count = 0
    total_features = []
    labels = []
    paths = []
    with open(features_path, newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        
        for i, row in enumerate(csv_reader):
            if i == 0:
                continue # Skip header row
            # Skip every 10 rows after each row
            # if i % 10 == 0:
            #     continue
            filepath = row[0]
            if exclude and filepath == exclude:
                continue # Exclude shape from returned features list if specified
            label = row.pop(1)
            row.pop(0)
            features = [float(feature.replace(" ", "")) for feature in row if feature != " "]
            total_features.append(features)
            labels.append(label)
            paths.append(filepath)
    
    return np.array(total_features), labels


# def get_all_features(features_path):
#     df = pd.read_csv(features_path)

#     # Skip the first column and row (header)
#     df = df.iloc[:, 1:]
#     df = df[1:]

#     # Create an array to store features and category labels
#     categories = []
#     features = []

#     # Skip 5 rows after each row and append the data to the features array
#     for i in range(0, len(df)):
#         category = df.iloc[i:i+1, 0:1]
#         print(category)
#         categories.append(category)
#         row = df.iloc[i:i+1, 1:]
#         features.append(row.values)
#         # Skip 10 meshes after each row
#         i += 10

#     return features, categories



def main():
    # Parameters
    features_path = "csvs/feature_extraction.csv"
    tsne_no_components = 2
    tsne_perplexity = 3

    # Load feature vectors for db shapes
    features_total, labels = get_all_features(features_path)

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
    # for x, y in zip(features_embedded[:,0], features_embedded[:,1]):
    #     ax.annotate(path.replace("/m", "-").replace(".obj", ""), (x, y))

    plt.savefig('2D_meshes.png')


if __name__ == "__main__":
    main()