import os
import random

import colorcet as cc
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE


def get_all_features(features_path: str) -> tuple:
    # Reduced version of Rorschach.querying.query.get_all_features
    if not os.path.exists(features_path):
        raise Exception(f"\nThe '{features_path}' file does not exist.")

    df = pd.read_csv(features_path)

    categories = df["category"].values
    features = df.drop(["filename", "category"], axis=1).astype(float).values

    return features, categories


def perform_tsne(fp_features: str, tsne_no_components: int = 2, tsne_perplexity: int = 10) -> tuple:
    # Load feature vectors for db shapes
    features, categories = get_all_features(fp_features)

    # Sort by category
    idx = np.argsort(categories)
    features = features[idx]
    categories = categories[idx]

    # Set NaN values to 0
    features = np.nan_to_num(features, nan=0, posinf=0, neginf=0)

    # Embed feature vectors in lower-dimensional space using T-distributed Stochastic Neighbor Embedding (t-SNE)
    features_embedded = TSNE(n_components=tsne_no_components, perplexity=tsne_perplexity, random_state=42).fit_transform(features)

    return features_embedded, categories


def reduce_data(features_embedded, categories, i, j, n, seed):
    # Get shuffled indices
    np.random.seed(seed)
    shuffled_indices = np.random.permutation(len(features_embedded))

    # Shuffle both arrays using the shuffled indices
    features_embedded = features_embedded[shuffled_indices]
    categories = categories[shuffled_indices]

    # Convert arrays to pandas DataFrame for easier manipulation
    df = pd.DataFrame({'features_embedded': features_embedded.tolist(), 'categories': categories})

    # Get unique categories
    unique_categories = df['categories'].unique()

    reduced_features = []
    reduced_categories = []

    # Select all categories between i and i+j
    for cat in unique_categories[i:i+j]:
        # Select n instances for each category
        selected_data = df[df['categories'] == cat][:n]

        # Retrieve the coordinates for each instance and store them in a nested list
        coords = [instance for instance in selected_data['features_embedded']]

        # Append the selected instances and corresponding categories
        reduced_features.extend(coords)
        reduced_categories.extend([cat] * len(coords))

    # Convert the nested list of coordinates back to a NumPy array
    reduced_features = np.array(reduced_features)
    reduced_categories = np.array(reduced_categories)

    return reduced_features, reduced_categories


def main_plot(features_embedded: np.array, categories: np.array, fp_out: str, i: int = 0, j: int = 69, n: int = 6, reduce: bool = False) -> None:
    # Color palette with n distinct colors
    palette = sns.color_palette(cc.glasbey, n_colors=j)

    # Reduce the number of data instances if desired
    if reduce:
        # Generate random seed
        seed = random.randint(0, 1000)
        # Update filepath
        fp_out = f"./figures/step5/2D_meshes({seed}).png"
        # Take a subset of the data (i.e. reduce the data to a specific amount)
        features_embedded, categories = reduce_data(features_embedded, categories, i, j, n, seed)

        # Set the figure size
        plt.figure(figsize=(9, 7))

        # Use Seaborn for plotting
        ax = sns.scatterplot(x=features_embedded[:, 0],
                             y=features_embedded[:, 1],
                             hue=categories,
                             palette=palette,
                             s=120,
                             legend=False)

        # Annotate each point with the category name
        for i, txt in enumerate(categories):
            ax.annotate(txt, (features_embedded[i, 0], features_embedded[i, 1]), textcoords="offset points", xytext=(5, 5), ha='right')

    else:
        # Set the figure size
        plt.figure(figsize=(22, 18))

        # Color palette with n distinct colors
        palette = sns.color_palette(cc.glasbey, n_colors=69)

        # Use Seaborn for plotting
        ax = sns.scatterplot(x=features_embedded[:, 0],
                             y=features_embedded[:, 1],
                             hue=categories,
                             palette=palette,
                             s=35)

        plt.legend(title="Categories", loc="lower left", fontsize="9")

    # Remove top and right spines
    sns.despine()

    # Add labels
    plt.xlabel("t-SNE component 1")
    plt.ylabel("t-SNE component 2")
    plt.tight_layout()

    # Save the plot
    plt.savefig(fp_out, dpi=300)


def interactive_plot(features_embedded: np.array, categories: np.array, n: int = 7) -> None:
    # Create a DataFrame
    data = pd.DataFrame({'tsne_component_1': features_embedded[:, 0],
                         'tsne_component_2': features_embedded[:, 1],
                         'category': categories})

    # Plotly scatter plot
    fig = px.scatter(data,
                     x='tsne_component_1',
                     y='tsne_component_2',
                     color='category',
                     title='Interactive Scatterplot',
                     opacity=0.7)

    fig.update_traces(marker=dict(size=10))

    def update_selected_points(trace, points, selector):
        inds = points.point_inds if points else []
        selected_category = data.loc[inds[0]]['category'] if inds else None

        if selected_category is not None:
            mask = data['category'] == selected_category
            fig.update_traces(selectedpoints=inds)
            fig.update_traces(marker=dict(color=np.where(mask, data['category'], 'lightgray'), opacity=np.where(mask, 0.7, 0.1)))

    fig.data[0].on_click(update_selected_points)

    fig.update_layout(xaxis_title='t-SNE component 1', yaxis_title='t-SNE component 2')

    fig.show()


if __name__ == "__main__":
    # Generate random seed between 0 and 1000
    fp_features = "./Rorschach/feature_extraction/features_normalized.csv"
    fp_out = "./figures/step5/2D_meshes_all.png"

    # Parameters
    tsne_no_components = 2
    # Perplexity value / Sigma value (should be between 30-50 according to Alex)
    # It accounts for the number of nearest neighbours that needs to be preserved after dim. reduction
    tsne_perplexity = 20

    # Perform t-SNE on all features and plot the 2-dimensional results (both as a whole and as an interactive plot)
    features_embedded, categories = perform_tsne(fp_features, tsne_no_components, tsne_perplexity)

    reduce = True

    # Plot all results or a specific subset
    if reduce:
        for _ in range(6):
            # Select only a couple of categories and instances per chosen category
            init = random.randint(0, 65)
            n_categories = 4
            n_instances = 6
            main_plot(features_embedded, categories, fp_out, i=init, j=n_categories, n=n_instances, reduce=reduce)
    else:
        main_plot(features_embedded, categories, fp_out)

    # interactive_plot(features_embedded, categories, n)
