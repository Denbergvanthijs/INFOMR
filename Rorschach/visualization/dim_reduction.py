import csv
import os
import numpy as np
import seaborn as sns
import colorcet as cc
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE



def get_features(features_path: str) -> tuple:
    mesh_paths = []
    categories = []
    features = []

    # Check if the file exists
    if not os.path.isfile(features_path):
        raise FileNotFoundError(f"Features file not found at {features_path}")

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
                features.append(row[2:])

    return mesh_paths, categories, np.array(features).astype(float)


def main(fp_features: str, fp_save: str, tsn_no_components: int = 2, tsne_perplexity: int = 10, i: int = 7) -> None:
    # Load feature vectors for db shapes
    mesh_paths, categories, features = get_features(fp_features)

    # Set NaN values to 0
    features = np.nan_to_num(features, nan=0, posinf=0, neginf=0)

    # Embed feature vectors in lower-dimensional space using T-distributed Stochastic Neighbor Embedding (t-SNE)
    features_embedded = TSNE(n_components=tsne_no_components, perplexity=tsne_perplexity, random_state=42).fit_transform(features)

    # categories = categories[:i] + categories[500:500+i] + categories[1000:1000+i] + categories[1500:1500+i]
    # to_stack = (features_embedded[:i], features_embedded[500:500+i], features_embedded[1000:1000+i], features_embedded[1500:1500+i])
    # features_embedded = np.vstack(to_stack)

    # Set the figure size
    plt.figure(figsize=(20, 17))

    # Color palette with 69 distinct colors
    palette = sns.color_palette(cc.glasbey, n_colors=69)

    # Use Seaborn for plotting
    sns.scatterplot(
        x=features_embedded[:, 0],
        y=features_embedded[:, 1],
        hue=categories,  
        palette=palette, 
        s=20
    )

    # Remove top and right spines
    sns.despine()

    plt.xlabel("t-SNE component 1")
    plt.ylabel("t-SNE component 2")
    plt.legend(title='Categories', loc='lower left', fontsize='8')
    plt.tight_layout()

    # Save the plot
    plt.savefig(fp_save, dpi=300)


if __name__ == "__main__":
    fp_features = "./Rorschach/feature_extraction/features_normalized.csv"
    fp_out = "./figures/step5/2D_meshes_all.png"

    # Parameters
    tsne_no_components = 2
    # Perplexity value / Sigma value (should be between 30-50 according to Alex)
    # It accounts for the number of nearest neighbours that needs to be preserved after dim. reduction
    tsne_perplexity = 25

    # Select only a couple of categories for visualization purposes
    n_categories = 7

    main(fp_features, fp_out, tsne_no_components, tsne_perplexity, n_categories)
