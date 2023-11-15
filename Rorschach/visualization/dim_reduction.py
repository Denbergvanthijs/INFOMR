import os
import numpy as np
import pandas as pd
import seaborn as sns
import colorcet as cc
import plotly.express as px
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


def main_plot(features_embedded: np.array, categories: np.array, fp_out: str, n: int = 7) -> None:
    to_stack = (categories[:n], categories[501:501+n], categories[1000:1000+n], categories[1500:1500+n])
    categories = np.vstack(to_stack).flatten()
    to_stack = (features_embedded[:n], features_embedded[501:501+n], features_embedded[1000:1000+n], features_embedded[1500:1500+n])
    features_embedded = np.vstack(to_stack)

    # Set the figure size
    plt.figure(figsize=(10, 8))

    # Color palette with 69 distinct colors
    palette = sns.color_palette(cc.glasbey, n_colors=4)

    # Use Seaborn for plotting
    sns.scatterplot(x=features_embedded[:, 0],
                    y=features_embedded[:, 1],
                    hue=categories,
                    palette=palette,
                    s=40)

    # Remove top and right spines
    sns.despine()

    plt.xlabel("t-SNE component 1")
    plt.ylabel("t-SNE component 2")
    plt.legend(title="Categories", loc="lower left", fontsize="10")
    plt.tight_layout()

    # Save the plot
    plt.savefig(fp_out, dpi=300)


def interactive_plot(features_embedded: np.array, categories: np.array, n: int = 7) -> None:
    # Create a DataFrame
    data = pd.DataFrame({
        'tsne_component_1': features_embedded[:, 0],
        'tsne_component_2': features_embedded[:, 1],
        'category': categories
    })

    # Plotly scatter plot
    fig = px.scatter(
        data,
        x='tsne_component_1',
        y='tsne_component_2',
        color='category',
        title='Interactive Scatterplot',
        opacity=0.7
    )

    fig.update_traces(marker=dict(size=10))

    def update_selected_points(trace, points, selector):
        inds = points.point_inds if points else []
        selected_category = data.loc[inds[0]]['category'] if inds else None

        if selected_category is not None:
            mask = data['category'] == selected_category
            fig.update_traces(selectedpoints=inds)
            fig.update_traces(marker=dict(color=np.where(mask, data['category'], 'lightgray'), opacity=np.where(mask, 0.7, 0.1)))

    fig.data[0].on_click(update_selected_points)

    fig.update_layout(
        xaxis_title='t-SNE component 1',
        yaxis_title='t-SNE component 2'
    )

    fig.show()


if __name__ == "__main__":
    fp_features = "./Rorschach/feature_extraction/features_normalized.csv"
    fp_out = "./figures/step5/2D_meshes_all.png"

    # Parameters
    tsne_no_components = 2
    # Perplexity value / Sigma value (should be between 30-50 according to Alex)
    # It accounts for the number of nearest neighbours that needs to be preserved after dim. reduction
    tsne_perplexity = 25

    # Select only a couple of categories for visualization purposes
    n = 7

    # Perform t-SNE on all features and plot the 2-dimensional results (both as a whole and as an interactive plot)
    features_embedded, categories = perform_tsne(fp_features, tsne_no_components, tsne_perplexity)
    main_plot(features_embedded, categories, fp_out, n)
    # interactive_plot(features_embedded, categories, n)
