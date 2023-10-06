import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_a3(data: np.ndarray, n_bins: int = 10) -> None:
    # Plot 16 linecharts on top of each other
    fig, ax = plt.subplots()

    x_range = np.arange(0.5, n_bins + 0.5)  # To center the lines in the middle of each bin
    for i in range(data.shape[0]):
        sns.lineplot(x=x_range, y=data[i], ax=ax, color="black", alpha=0.2)

    # X ticks in range 0 to 180 degrees
    plt.xticks(range(n_bins + 1), [f"{i * 180 / n_bins:.0f}" for i in range(n_bins + 1)])
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Frequency")
    plt.xlim(0, n_bins)
    plt.title("A3 histogram")

    plt.tight_layout()
    plt.savefig("./figures/feat_ex/a3_hist.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    # Load data
    data = np.loadtxt("./csvs/feature_extraction.csv", delimiter=",", skiprows=1, usecols=range(2, 12))
    n_bins = 10

    plot_a3(data, n_bins)
