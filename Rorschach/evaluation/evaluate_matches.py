import pandas as pd
import matplotlib.pyplot as plt


def calculate_acc(fp):
    df_all_matches = pd.read_csv(fp)
    df_all_matches["correct"] = df_all_matches["query_category"] == df_all_matches["match_category"]

    # Calculate accuracy per category
    df_grouped = df_all_matches.groupby("query_category")["correct"].mean()
    # df_grouped = df_grouped.sort_values(ascending=False)

    return df_grouped


if __name__ == "__main__":
    fps = ["./Rorschach/evaluation/data/collect_neighbours_knn.csv",
           "./Rorschach/evaluation/data/collect_neighbours_manhattan.csv",
           "./Rorschach/evaluation/data/collect_neighbours_euclidean.csv",
           "./Rorschach/evaluation/data/collect_neighbours_cosine.csv",
           "./Rorschach/evaluation/data/collect_neighbours_emd.csv"]
    # columns: query_filepath,query_category,match_filepath,match_category,distance

    accs = [calculate_acc(fp) for fp in fps]  # Calculate accuracy per distance function

    fig, axes = plt.subplots(1, 5, figsize=(20, 5), sharey=True)
    for i, (fp, acc) in enumerate(zip(fps, accs)):
        axes[i].bar(acc.index, acc.values)
        axes[i].set_xticklabels(acc.index, rotation=90)

        text = fp.split("/")[-1].split(".")[0].split("_")[-1]  # knn, manhattan, euclidean, cosine, emd
        text = text.title() if len(text) > 3 else text.upper()  # KNN and EMD, others Titlecase
        axes[i].set_title(text)

    plt.suptitle("Accuracy per category and per distance function")
    plt.tight_layout()
    plt.show()
