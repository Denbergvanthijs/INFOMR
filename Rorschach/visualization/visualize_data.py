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
def boxplot(mesh_info, column: str, fp_save: str) -> None:
    # Load vertice and face counts
    data = mesh_info[column].values
    total_meshes = len(data)  # Total number of meshes

    # Boxplot showing number of vertices and faces
    fig, ax = plt.subplots(figsize=(6, 8))
    # X-axis labels off
    plt.boxplot(data, labels=[""])
    plt.title(f"Boxplot of number of {column.lower()} per mesh (n={total_meshes})")
    plt.ylabel(f"Number of {column.lower()}")

    ax.set_axisbelow(True)  # Put grid behind bars
    ax.grid(axis="y")
    plt.ylim(0, None)

    plt.tight_layout()
    plt.savefig(fp_save, dpi=300)  # Dont save as eps, since opacity will be lost otherwise

    # Reset plot for future plotting
    plt.clf()


# Function to create 2D histogram of either vertice or face count
def histogram2D(mesh_info, column, fp_save, n_bins: int = 15) -> None:
    # Calculate the mean of the chosen variable (i.e. "Vertices" or "Faces")
    mean_value = mesh_info[column].mean()
    total_meshes = len(mesh_info)  # Total number of meshes

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot histogram based on variable choice
    sns.histplot(data=mesh_info, color=UU_YELLOW, x=column, bins=n_bins, kde=False, ax=ax)
    plt.title(f"Histogram of number of {column.lower()} per mesh (n={total_meshes})")
    plt.xlabel(f"Number of {column.lower()}")
    plt.ylabel("Number of meshes")

    # Add a vertical line at the mean
    plt.axvline(mean_value, color=UU_RED, linestyle="dashed", label=f"{mean_value:.0f}")
    plt.legend(title=f"Mean number of {column.lower()}", loc="upper right")

    ax.set_axisbelow(True)  # Put grid behind bars
    ax.grid(axis="y")

    # Left limit of x-axis set to 0
    plt.xlim(0, None)

    plt.tight_layout()
    plt.savefig(fp_save, dpi=300)  # Dont save as eps, since opacity will be lost otherwise

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


def class_distribution(mesh_info: pd.DataFrame, fp_out: str, top_n: int = 10) -> None:
    counts = mesh_info["Class"].value_counts()
    counts_len = len(counts)  # Total number of categories
    counts_mean = counts.mean()  # Mean number of shapes per category for all categories

    # Only top n classes
    counts = counts[:top_n]

    # Plot seaborn barplot, class names on y-axis
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=counts.values, y=counts.index, color=UU_YELLOW, ax=ax, edgecolor="black")
    plt.title(f"Class distribution of top {top_n} out of {counts_len} categories")
    plt.xlabel("Number of meshes in category")
    plt.ylabel("")

    # Add counts to each bar
    for c in ax.containers:
        ax.bar_label(c, fmt=" %.0f", color="black")

    # Plot average count
    plt.axvline(counts_mean, color=UU_RED, linestyle="dashed", label=f"{counts_mean:.2f}")
    plt.legend(title="Mean number of meshes\nper category", loc="lower right")

    plt.tight_layout()
    plt.savefig(fp_out, dpi=300)

    # Reset plot for future plotting
    plt.clf()


def class_histogram(mesh_info, fp_save, every_n: int = 20) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    data = mesh_info["Class"].value_counts().values
    bins = np.arange(0, data.max() + every_n, every_n)

    ax.hist(data, bins=bins, color=UU_YELLOW, edgecolor="black", range=(0, data.max() + every_n))
    ax.set_xticks(bins)  # Set x-axis ticks
    ax.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)  # Remove y ticks

    # Add counts to each bar
    for c in ax.containers:
        ax.bar_label(c, fmt="%.0f categories", color="black")

    plt.title(f"Histogram of number of meshes in each category (|C|={len(data)})")
    plt.xlabel("Number of meshes in category")

    # Limit x-axis to 0
    plt.xlim(0, bins[-1])
    plt.tight_layout()
    plt.savefig(fp_save, dpi=300)

    # Reset plot for future plotting
    plt.clf()


def hist_barycenter(mesh_info: pd.DataFrame, mesh_info_normalized: pd.DataFrame, column: str,
                    fp_save: str, hist_range: tuple = (0, 1), sharex: bool = False, sharey: bool = False) -> None:
    # Compute histograms
    offset_bary_count, bins_bary = np.histogram(mesh_info[column], bins="sqrt", range=hist_range)
    offset_bary_count_norm, bins_bary_norm = np.histogram(mesh_info_normalized[column], bins="sqrt")

    # Change data to percentages
    offset_bary_count = offset_bary_count / offset_bary_count.sum() * 100
    offset_bary_count_norm = offset_bary_count_norm / offset_bary_count_norm.sum() * 100

    print(f"Max barycenter offset: {mesh_info['Barycenter offset'].max()}")
    print(f"Max barycenter offset (normalized): {mesh_info_normalized['Barycenter offset'].max()}")

    fig, axes = plt.subplots(1, 2, sharex=sharex, sharey=sharey)
    axes[0].hist(bins_bary[:-1], bins_bary, weights=offset_bary_count, color=UU_YELLOW, edgecolor="black")
    axes[0].set_title("Before normalization")
    axes[0].set_ylabel("Percentage")
    axes[0].set_xlabel("\nBarycenter Squared Distance\nfrom origin")
    axes[0].set_xlim(0, 1)

    axes[1].hist(bins_bary_norm[:-1], bins_bary_norm, weights=offset_bary_count_norm, color=UU_YELLOW, edgecolor="black")
    axes[1].set_title("After normalization")
    axes[1].set_ylabel("Percentage")
    axes[1].set_xlabel("\nBarycenter Squared Distance\nfrom origin")
    axes[1].set_xlim(0, None)

    plt.tight_layout()
    plt.savefig(fp_save, dpi=300)

    plt.clf()


def hist_max_dim(mesh_info: pd.DataFrame, mesh_info_normalized: pd.DataFrame, column: str, fp_save: str) -> None:
    # Compute max dim histograms
    max_dim_count, bins_max_dim = np.histogram(mesh_info[column], bins="sqrt", range=(0, 20))
    max_dim_count_norm, bins_max_dim_norm = np.histogram(mesh_info_normalized[column], bins="sqrt")

    # Change data to percentages
    max_dim_count = max_dim_count / max_dim_count.sum() * 100
    max_dim_count_norm = max_dim_count_norm / max_dim_count_norm.sum() * 100

    fig, axes = plt.subplots(1, 2)
    axes[0].hist(bins_max_dim[:-1], bins_max_dim, weights=max_dim_count, color=UU_YELLOW, edgecolor="black")
    axes[0].set_title("Before normalization")
    axes[0].set_ylabel("Percentage")
    axes[0].set_xlabel("\nLength of longest dimension\nof bounding box")

    axes[1].hist(bins_max_dim_norm[:-1], bins_max_dim_norm, weights=max_dim_count_norm, color=UU_YELLOW, edgecolor="black")
    axes[1].set_title("After normalization")
    axes[1].set_ylabel("Percentage")
    axes[1].set_xlabel("\nLength of longest dimension\nof bounding box")

    plt.tight_layout()
    plt.savefig(fp_save, dpi=300)
    # plt.show()

    plt.clf()


def hist_before_after(mesh_info, mesh_info_normalized, column: str, fp_save,
                      binrange: tuple = None, sharex: bool = True, sharey: bool = True) -> None:
    data_before = mesh_info[column].values
    data_after = mesh_info_normalized[column].values

    fig, axes = plt.subplots(1, 2, sharex=sharex, sharey=sharey)
    sns.histplot(data=data_before, color=UU_YELLOW, ax=axes[0], stat="percent", edgecolor="black", bins="sqrt", binrange=binrange)
    sns.histplot(data=data_after, color=UU_YELLOW, ax=axes[1], stat="percent", edgecolor="black", bins="sqrt", binrange=binrange)

    axes[0].set_ylabel("Percentage")
    axes[0].set_xlabel(f"\n{column}")
    axes[1].set_xlabel(f"\n{column}")

    # Rotate x-axis labels if column is Vertices or Faces
    if column in ["Vertices", "Faces"]:
        axes[0].tick_params(axis="x", rotation=45)
        axes[1].tick_params(axis="x", rotation=45)

        axes[0].set_title("Before resampling")
        axes[1].set_title("After resampling")
    else:
        axes[0].set_title("Before normalization")
        axes[1].set_title("After normalization")

    plt.tight_layout()
    plt.savefig(fp_save, dpi=300)
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

    # 2.2 statistics over whole dataset
    histogram2D(mesh_info_raw, "Vertices", "./figures/step2/before_processing/hist_vertices.png", n_bins=15)
    histogram2D(mesh_info_raw, "Faces", "./figures/step2/before_processing/hist_faces.png", n_bins=15)
    boxplot(mesh_info_raw, column="Vertices", fp_save="./figures/step2/before_processing/boxplot_vertices.png")
    boxplot(mesh_info_raw, column="Faces", fp_save="./figures/step2/before_processing/boxplot_faces.png")
    class_distribution(mesh_info_raw, "./figures/step2/before_processing/class_distribution.png", top_n=10)
    class_histogram(mesh_info_raw, "./figures/step2/before_processing/class_histogram.png", every_n=20)

    # 2.5
    hist_barycenter(mesh_info, mesh_info_normalized, "Barycenter offset", "./figures/step2/after_processing/hist_bary.png")
    hist_max_dim(mesh_info, mesh_info_normalized, "Max dim", "./figures/step2/after_processing/hist_max_dim.png")
    hist_before_after(mesh_info, mesh_info_normalized, "Principal comp error", "./figures/step2/after_processing/hist_pca.png")
    hist_before_after(mesh_info, mesh_info_normalized, "SOM error", "./figures/step2/after_processing/hist_som.png")

    # Uses mesh info raw since only these two columns are present in the raw data
    hist_before_after(mesh_info_raw, mesh_info_normalized, "Vertices",
                      "./figures/step2/after_processing/hist_vertices.png", binrange=(0, 100_000))
    hist_before_after(mesh_info_raw, mesh_info_normalized, "Faces",
                      "./figures/step2/after_processing/hist_faces.png", binrange=(0, 100_000))

    # Old functions
    # histogram3D(mesh_info_raw)
    # boxplot_before_after(mesh_info, mesh_info_normalized, "Max dim")
    # print("Max dim before / after normalization:")
    # print(f"Median: {np.median(mesh_info['Max dim'])} / {np.median(mesh_info_normalized['Max dim'])}")
    # print(f"SD: {np.std(mesh_info['Max dim'])} / {np.std(mesh_info_normalized['Max dim'])}")
