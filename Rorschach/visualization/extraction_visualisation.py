import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.contrib import tzip


def plot_feat_hist(data: np.ndarray, title: str, feat: str, x_label: str, x_ticks_label: list, n_bins: int = 10) -> None:
    # Plot all linecharts on top of each other
    fig, ax = plt.subplots()

    x_range = np.arange(0.5, n_bins + 0.5)  # To center the lines in the middle of each bin
    for i in range(data.shape[0]):
        sns.lineplot(x=x_range, y=data[i], ax=ax, color="black", alpha=0.2)

    # X ticks in range 0 to n_bins + 1
    plt.xticks(range(n_bins + 1), x_ticks_label)
    plt.xlabel(x_label)
    plt.ylabel("Frequency")
    plt.xlim(0, n_bins)
    plt.ylim(0, None)
    plt.title(title)

    plt.tight_layout()
    plt.savefig(f"./figures/feat_ex/feat_hist_{feat.replace(' ', '_').lower()}.png", dpi=300)
    # plt.show()


def plot_hist_grid(df, fp_save: str, n_classes: int = 4, n_bins: int = 10) -> None:
    # Select the n_classes most popular classes
    df = df[df["category"].isin(df["category"].value_counts().index[:n_classes])]
    print(df["category"].value_counts())

    # Create a grid of subplots
    fig, axes = plt.subplots(nrows=5, ncols=n_classes, figsize=(10, 6), sharex=True, sharey=False)

    # Iterate over all classes in the dataset
    for i, category in enumerate(df["category"].unique()):
        # Select the rows in the dataframe that belong to the current class
        df_class = df[df["category"] == category]

        # Iterate over all features
        for j, feat in enumerate(["a3", "d1", "d2", "d3", "d4"]):
            plt.sca(axes[j, i])  # Set current subplot
            cols = [f"{feat}_{i}" for i in range(n_bins)]  # Column names

            # Plot all linecharts on top of each other
            x_range = np.arange(n_bins)
            for k in range(df_class.shape[0]):
                plt.plot(x_range, df_class[cols].iloc[k], color="black", alpha=0.2)

    # Set x labels
    for i, (category, value) in enumerate(df["category"].value_counts().items()):
        plt.sca(axes[-1, i])
        plt.xlabel(f"{category}\n({value} meshes)")

    # Set y labels
    for i, feat in enumerate(["A3", "D1", "D2", "D3", "D4"]):
        plt.sca(axes[i, 0])
        plt.ylabel(feat, rotation=0, labelpad=20)

    # Disable all ticks
    for ax in axes.ravel():
        ax.tick_params(axis="both", which="both", bottom=False, left=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, n_bins)

    plt.tight_layout()
    plt.savefig(fp_save, dpi=300)
    plt.show()


if __name__ == "__main__":
    # Load data
    fp_data = "./Rorschach/feature_extraction/features.csv"
    n_bins = 10
    n_iter = 1_000
    n_classes = 4

    # Load csv in pandas dataframe
    df = pd.read_csv(fp_data, delimiter=",")

    # Plot grid before filtering
    plot_hist_grid(df, "./figures/step3/feat_hist_grid.png", n_classes, n_bins=n_bins)

    # Old code/unused figures
    # Round to 1 decimal place for floating point errors like 0.300004
    # x_ranges = [range(0, 181, 18)] + 4 * [np.arange(0, 1.1, 0.1).round(1)]
    # features = ["a3", "d1", "d2", "d3", "d4"]
    # labels = ["Angle (degrees)", "Distance (unit size)", "Distance (unit size)",
    #           "Square root distance (unit size)", "Cube root distance (unit size)"]

    # for feat, label, x_range in tzip(features, labels, x_ranges):
    #     data = df.filter(regex=f"{feat}_").to_numpy()
    #     title = f"{feat.upper()} histogram of all classes combined ({n_bins} bins, {n_iter} iterations)"
    #     plot_feat_hist(data, title, feat, label, x_range, n_bins=n_bins)
