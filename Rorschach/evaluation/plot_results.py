import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

UU_YELLOW = "#FFCD00"
UU_RED = "#C00A35"
UU_CREME = "#FFE6AB"
UU_ORANGE = "#F3965E"
UU_BURGUNDY = "#AA1555"
UU_BROWN = "#6E3B23"
UU_BLUE = "#5287C6"
UU_PAL = sns.color_palette([UU_YELLOW, UU_RED, UU_CREME, UU_ORANGE, UU_BURGUNDY, UU_BROWN, UU_BLUE])


def plot_perclass_metrics(data_dict, metric, distance_function, k=None):
    labels, values = zip(*data_dict.items())
    k_str = k if k is not None else "|c|"
    metric_str = metric.lower().replace(" ", "_")

    fig, ax = plt.subplots()
    ax = plt.bar(labels, values, color=UU_YELLOW, width=0.7, edgecolor="black")
    plt.title(f"{metric.title()} per category for the {distance_function} distance function (k={k_str})")

    plt.xticks(rotation=90)
    plt.ylabel(metric.title())

    plt.tight_layout()
    plt.savefig(f"./figures/step6/perclass/perclass_{metric_str}_k{k}_{distance_function}.png", dpi=300)
    # plt.show()


def plot_overall_metric(df, metric):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax = sns.barplot(x="Distance function", y=metric, data=df, ax=ax, palette=UU_PAL, edgecolor="black", hue="k")
    # plt.title(f"{metric.title()} per distance function")
    plt.ylabel(metric.title())

    # Change legend
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, ["k=3", "k=|c|"], title="Top k meshes to consider", bbox_to_anchor=(1.05, 1))

    ax.set_axisbelow(True)  # Put grid behind bars
    ax.grid(axis="y")

    plt.tight_layout()
    plt.savefig(f"./figures/step6/overall_{metric.lower().replace(' ', '_')}.png", dpi=300)
    # plt.show()


if __name__ == "__main__":
    fp_results = "./Rorschach/evaluation/data/overall_results.csv"
    df = pd.read_csv(fp_results)
    print(df.head())

    plot_overall_metric(df, "Precision")
    plot_overall_metric(df, "Recall")
    plot_overall_metric(df, "Accuracy")
    plot_overall_metric(df, "Sensitivity")
    plot_overall_metric(df, "Specificity")
    plot_overall_metric(df, "F1 score")
    plot_overall_metric(df, "F2 score")
