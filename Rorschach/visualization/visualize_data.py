import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Colors for visualizing (official UU colors from https://www.uu.nl/en/organisation/corporate-identity/brand-policy/colour)
UU_YELLOW = "#FFCD00"
UU_RED = "#C00A35"
UU_CREME = "#FFE6AB"
UU_ORANGE = "#F3965E"
UU_BURGUNDY = "#AA1555"
UU_BROWN = "#6E3B23"
UU_BLUE = "#5287C6"
UU_PAL = sns.color_palette([UU_YELLOW, UU_RED, UU_CREME, UU_ORANGE, UU_BURGUNDY, UU_BROWN, UU_BLUE])


# Function to calculate outliers (shapes outside of 4th quartile of boxplot)
def calc_outliers(data):
    # Calculate the quartiles and IQR
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1

    # Define the upper bound for outliers
    upper_bound = q3 + 1.5 * iqr
    print(upper_bound)

    # Count the number of outliers
    outliers = [x for x in data if x > upper_bound]
    num_outliers = len(outliers)
    print(num_outliers)


# Function to create boxplots of number of vertices and faces
def boxplot(mesh_info, column: str) -> None:
    # Load vertice and face counts
    data = mesh_info[column].values

    # Min and max number of faces and vertices
    # print(min(vertices), min(faces))      16, 16
    # print(max(vertices), max(faces))      98256, 129881

    # Number of outliers (shapes outside of 4th quartile)
    # calc_outliers(vertices)      upper whisker: 15032, num. outliers: 247
    # calc_outliers(faces)         upper whisker: 31986.5, num. outliers: 256

    # Boxplot showing number of vertices and faces
    fig = plt.figure(figsize=(6, 8))
    # X-axis labels off
    plt.boxplot(data, labels=[""])
    plt.title(f"Boxplot of number of {column.lower()}")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"./figures/boxplot_{column.lower()}.eps")

    # Reset plot for future plotting
    plt.clf()


# Function to create 2D histogram of either vertice or face count
def histogram2D(mesh_info, column, n_bins=15):
    # Calculate the mean of the chosen variable (i.e. "Vertices" or "Faces")
    mean_value = mesh_info[column].mean()

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot histogram based on variable choice
    sns.histplot(data=mesh_info, color=UU_BLUE, x=column, bins=n_bins, kde=True, ax=ax)
    plt.title(f"2D Histogram of number of {column}")
    plt.xlabel(f"Number of {column.lower()}")
    plt.ylabel("Frequency")

    # Add a vertical line at the mean
    plt.axvline(mean_value, color=UU_RED, linestyle="dashed", label=f"Mean ({mean_value:.2f})")
    plt.legend()

    plt.tight_layout()
    # Dont save as eps, since opacity will be lost otherwise
    plt.savefig(f"./figures/{column.lower()}_hist.png")

    # Reset plot for future plotting
    plt.clf()


# Function to create a 3D histogram of shapes in dataset
def histogram3D(mesh_info):
    # Load vertice and face counts
    vertices = mesh_info["Vertices"].values
    faces = mesh_info["Faces"].values

    # Create a 3D histogram using Seaborn
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="3d")

    n_bins = 15

    xedges, yedges = np.linspace(0, 100000, n_bins), np.linspace(0, 140000, n_bins)
    hist, xedges, yedges = np.histogram2d(vertices, faces, bins=(xedges, yedges), range=[[0, 100000], [0, 140000]])

    xpos, ypos = np.meshgrid(xedges[:-1] + 0.5 * np.diff(xedges), yedges[:-1] + 0.5 * np.diff(yedges))

    ax.bar3d(xpos.flatten(), ypos.flatten(), np.zeros_like(hist).flatten(),
             dx=np.diff(xedges)[0], dy=np.diff(yedges)[0], dz=hist.flatten(), color=UU_YELLOW, shade=True)

    ax.set_xlabel("Vertices", fontsize=11)
    ax.set_ylabel("Faces", fontsize=11)
    ax.set_zlabel("Frequency", fontsize=11)
    ax.set_title("3D Shape Histogram", fontsize=20)

    plt.savefig("./figures/3D_histogram.eps")

    # Reset plot for future plotting
    plt.clf()


def class_distribution(mesh_info: pd.DataFrame, top_n: int = 10, fp_out: str = "./figures/class_distribution.eps") -> None:
    counts = mesh_info["Class"].value_counts()
    counts_len = len(counts)
    counts_mean = counts.mean()

    # Only top 10 classes
    counts = counts[:top_n]

    # Plot seaborn barplot, class names on y-axis
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=counts.values, y=counts.index, color=UU_YELLOW, ax=ax)
    plt.title(f"Class distribution of top {top_n} out of {counts_len} classes")
    plt.xlabel("Number of shapes")
    plt.ylabel("Class")

    # Add counts to each bar
    for c in ax.containers:
        ax.bar_label(c, fmt=" %.0f", color=UU_RED)

    # Plot average count
    plt.axvline(counts_mean, color=UU_RED, linestyle="dashed", label=f"Mean ({counts_mean:.2f})")
    plt.legend()

    plt.tight_layout()
    plt.savefig(fp_out)

    # Reset plot for future plotting
    plt.clf()


def class_histogram(mesh_info, every_n: int = 20) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    data = mesh_info["Class"].value_counts().values
    bins = np.arange(0, data.max() + every_n, every_n)

    ax.hist(data, bins=bins, color=UU_YELLOW, edgecolor=UU_RED, range=(0, data.max() + every_n))
    ax.set_xticks(bins)  # Set x-axis ticks
    ax.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)  # Remove y ticks

    # Add counts to each bar
    for c in ax.containers:
        ax.bar_label(c, fmt="%.0f class(es)", color=UU_RED)

    plt.title("Histogram of number of shapes in each class")
    plt.xlabel("Number of shapes in class")
    plt.tight_layout()
    plt.savefig("./figures/class_histogram.eps")

    # Reset plot for future plotting
    plt.clf()


def barycenter_hist(mesh_info: pd.DataFrame, mesh_info_normalized: pd.DataFrame, column: str) -> None:
    # Compute barycenter offset histograms
    offset_bary_count, bins_bary = np.histogram(mesh_info[column], bins="sqrt", range=(0, 1))
    offset_bary_count_norm, bins_bary_norm = np.histogram(mesh_info_normalized[column], bins="sqrt")

    # Change data to percentages
    offset_bary_count = offset_bary_count / offset_bary_count.sum() * 100
    offset_bary_count_norm = offset_bary_count_norm / offset_bary_count_norm.sum() * 100

    print(f"Max offset: {mesh_info['Barycenter offset'].max()}")
    print(f"Max offset (norm): {mesh_info_normalized['Barycenter offset'].max()}")

    fig, axes = plt.subplots(1, 2)
    axes[0].hist(bins_bary[:-1], bins_bary, weights=offset_bary_count, color=UU_YELLOW, edgecolor=UU_RED)
    axes[0].set_title("Before Normalization")
    axes[0].set_ylabel("Percentage")
    axes[0].set_xlabel("\nBarycenter Squared Distance\nfrom Origin")

    axes[1].hist(bins_bary_norm[:-1], bins_bary_norm, weights=offset_bary_count_norm, color=UU_YELLOW, edgecolor=UU_RED)
    axes[1].set_title("After Normalization")
    axes[1].set_ylabel("Percentage")
    axes[1].set_xlabel("\nBarycenter Squared Distance\nfrom Origin")

    plt.tight_layout()
    plt.savefig("./figures/barycenter_histogram.eps")

    plt.clf()


def max_dim_hist(mesh_info: pd.DataFrame, mesh_info_normalized: pd.DataFrame, column: str) -> None:
    # Compute max dim histograms
    max_dim_count, bins_max_dim = np.histogram(mesh_info[column], bins="sqrt", range=(0, 20))
    max_dim_count_norm, bins_max_dim_norm = np.histogram(mesh_info_normalized[column], bins="sqrt")

    # Change data to percentages
    max_dim_count = max_dim_count / max_dim_count.sum() * 100
    max_dim_count_norm = max_dim_count_norm / max_dim_count_norm.sum() * 100

    max_dim_median = np.median(mesh_info[column])
    max_dim_sd = np.std(mesh_info[column])

    fig, axes = plt.subplots(1, 2)
    axes[0].hist(bins_max_dim[:-1], bins_max_dim, weights=max_dim_count, color=UU_YELLOW, edgecolor=UU_RED)
    axes[0].set_title("Before Normalization")
    axes[0].set_ylabel("Percentage")
    axes[0].set_xlabel("\nBounding Box Longest Dimension")

    axes[1].hist(bins_max_dim_norm[:-1], bins_max_dim_norm, weights=max_dim_count_norm, color=UU_YELLOW, edgecolor=UU_RED)
    axes[1].set_title("After Normalization")
    axes[1].set_ylabel("Percentage")
    axes[1].set_xlabel("\nBounding Box Longest Dimension")

    plt.tight_layout()
    plt.savefig(f"./figures/hist_{column.lower().replace(' ', '_')}_before_after.eps")
    # plt.show()

    plt.clf()


def hist_before_after(mesh_info, mesh_info_normalized, column: str, sharex: bool = True, sharey: bool = True) -> None:
    data_before = mesh_info[column].values
    data_after = mesh_info_normalized[column].values

    fig, axes = plt.subplots(1, 2, sharex=sharex, sharey=sharey)
    if column == "Vertices" or column == "Faces":
        sns.histplot(data=data_before, color=UU_YELLOW, ax=axes[0], stat="percent", binrange=(0,100000))
        sns.histplot(data=data_after, color=UU_YELLOW, ax=axes[1], stat="percent", binrange=(0,100000))
    else:
        sns.histplot(data=data_before, color=UU_YELLOW, ax=axes[0], stat="percent")
        sns.histplot(data=data_after, color=UU_YELLOW, ax=axes[1], stat="percent")

    axes[0].set_title("Before normalization")
    axes[0].set_ylabel("Percentage")
    axes[0].set_xlabel(f"\n{column}")

    axes[1].set_title("After normalization")
    axes[1].set_ylabel("Percentage")
    axes[1].set_xlabel(f"\n{column}")

    plt.tight_layout()
    plt.savefig(f"./figures/hist_{column.lower().replace(' ', '_')}_before_after.eps")
    # plt.show()

    plt.clf()


def boxplot_before_after(mesh_info, mesh_info_normalized, column: str) -> None:
    data_before = mesh_info[column].values
    data_after = mesh_info_normalized[column].values

    fig, axes = plt.subplots(1, 2)
    sns.boxplot(data=data_before, color=UU_YELLOW, ax=axes[0], showfliers=False)
    sns.boxplot(data=data_after, color=UU_YELLOW, ax=axes[1], showfliers=False)

    axes[0].set_title("Before normalization")
    axes[0].set_ylabel(f"\n{column}")

    axes[1].set_title("After normalization")
    axes[1].set_ylabel(f"\n{column}")

    plt.tight_layout()
    plt.savefig(f"./figures/boxplot_{column.lower().replace(' ', '_')}_before_after.eps")
    # plt.show()

    plt.clf()


if __name__ == "__main__":
    # Load mesh info from existing CSV file
    mesh_info_raw = pd.read_csv("./data/mesh_info.csv")
    mesh_info = pd.read_csv("./data_cleaned/mesh_info.csv")
    mesh_info_normalized = pd.read_csv("./data_normalized/mesh_info.csv")

    # 2.2
    histogram2D(mesh_info_raw, "Vertices")
    histogram2D(mesh_info_raw, "Faces")
    boxplot(mesh_info_raw, column="Vertices")
    boxplot(mesh_info_raw, column="Faces")
    histogram3D(mesh_info_raw)
    class_distribution(mesh_info_raw)
    class_histogram(mesh_info_raw)

    # 2.5
    barycenter_hist(mesh_info, mesh_info_normalized, "Barycenter offset")
    hist_before_after(mesh_info, mesh_info_normalized, "Principal comp error")
    hist_before_after(mesh_info, mesh_info_normalized, "SOM error")
    boxplot_before_after(mesh_info, mesh_info_normalized, "Max dim")
    print("Max dim before / after normalization:")
    print(f"Median: {np.median(mesh_info['Max dim'])} / {np.median(mesh_info_normalized['Max dim'])}")
    print(f"SD: {np.std(mesh_info['Max dim'])} / {np.std(mesh_info_normalized['Max dim'])}")
    max_dim_hist(mesh_info, mesh_info_normalized, "Max dim")
    hist_before_after(mesh_info_raw, mesh_info_normalized, "Vertices")
    hist_before_after(mesh_info_raw, mesh_info_normalized, "Faces")
