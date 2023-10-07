import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_feat_hist(data: np.ndarray, feat: str, x_label: str, x_ticks_label: list, n_bins: int = 10) -> None:
    # Plot 16 linecharts on top of each other
    fig, ax = plt.subplots()

    x_range = np.arange(0.5, n_bins + 0.5)  # To center the lines in the middle of each bin
    for i in range(data.shape[0]):
        sns.lineplot(x=x_range, y=data[i], ax=ax, color="black", alpha=0.2)

    # X ticks in range 0 to 180 degrees
    plt.xticks(range(n_bins + 1), x_ticks_label)
    plt.xlabel(x_label)
    plt.ylabel("Frequency")
    plt.xlim(0, n_bins)
    plt.title(f"{feat} histogram")

    plt.tight_layout()
    plt.savefig(f"./figures/feat_ex/feat_hist_{feat}.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    # Load data
    data = np.loadtxt("./csvs/feature_extraction.csv", delimiter=",", skiprows=1, usecols=range(2, 22))
    n_bins = 10

    a3 = data[:, :n_bins]
    d1 = data[:, n_bins:]

    plot_feat_hist(a3, "A3", "Angle (degrees)", range(0, 181, 18))
    plot_feat_hist(d1, "D1", "Distance (unit size)", np.arange(0, 1.1, 0.1).round(1), n_bins=10)
