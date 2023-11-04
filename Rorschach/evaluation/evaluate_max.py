from collections import defaultdict
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

UU_YELLOW = "#FFCD00"


def get_query_results(results_path):
    # columns: query_filepath,query_category,match_filepath,match_category,distance
    df = pd.read_csv(results_path, sep=",")

    # For each query, get a list of the matched categories
    query_results = {}
    ground_truth = {}
    for query_filepath in df["query_filepath"].unique():
        # The categories of all matches for the current query
        query_results[query_filepath] = list(df[df["query_filepath"] == query_filepath]["match_category"])
        # The category of the current query, select single value from df
        ground_truth[query_filepath] = df[df["query_filepath"] == query_filepath]["query_category"].iloc[0]

    return query_results, ground_truth


def plot_perclass_metrics(data_dict, metric, k=None):
    # Plot histogram
    labels = list(data_dict.keys())
    values = list(data_dict.values())
    # items = [(label, value) for label, value in zip(labels, values)]
    # items.sort(key=lambda x: x[1])
    # labels, values = zip(*items)

    plt.bar(labels, values, color=UU_YELLOW, width=0.7, edgecolor="black")
    plt.xticks(rotation=90)
    plt.ylabel(metric.title())
    plt.title(f"{metric.title()} per category")

    plt.tight_layout()
    plt.savefig(f"./Rorschach/evaluation/plots/perclass_{metric.lower().replace(' ', '_')}_k{k}.png")
    plt.show()


def calculate_perclass(query_results_path: str, plot_type: str, k: int = None):
    query_results, ground_truth = get_query_results(query_results_path)

    # Compute metrics
    TPs = defaultdict(list)
    FPs = defaultdict(list)
    TNs = defaultdict(list)
    FNs = defaultdict(list)
    database_size = len(query_results)

    for i, (y_pred, query_class) in enumerate(zip(list(query_results.values()), ground_truth.values())):
        if k is not None:  # Limit to the top k results, results are already sorted by distance
            y_pred = y_pred[:k]

        query_size = len(y_pred)

        # Get true positives/negatives and false positives/negatives
        TP = y_pred.count(query_class)          # Correctly labelled as 'member of query class'
        # Incorrectly labelled as 'member of query class' (i.e. all returned shapes that are not part of the query class)
        FP = query_size - TP
        # Correctly labelled as 'NOT a member of query class' (i.e. all shapes in the database not part of query class that were not returned)
        TN = database_size - query_size - FP
        # Incorrectly labelled as 'NOT a member of query class' (i.e. all shapes in the database that are
        # a part of the query class but were not returned)
        FN = query_size - TP

        # Store performance metric results
        for metric, value in zip([TPs, FPs, TNs, FNs], [TP, FP, TN, FN]):
            metric[query_class].append(value)

    return TPs, FPs, TNs, FNs


def calculate_metrics(TPs, FPs, TNs, FNs):
    d = {}
    for key in TPs.keys():  # Each category has its TP, FP, TN, FN values stored per queried shape
        TP = np.array(TPs[key])
        FP = np.array(FPs[key])
        TN = np.array(TNs[key])
        FN = np.array(FNs[key])

        # Compute performance metrics row wise, thus for all queried shapes of a category in one go
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        sensitivity = recall
        specificity = TN / (TN + FP)
        f1_score = 2 * ((precision * recall) / (precision + recall))

        results = [precision, recall, accuracy, sensitivity, specificity, f1_score]
        results = [np.mean(r) for r in results]  # Average over all shapes in category

        d[key] = results

    return d


def calculate_overall(perclass_score: dict) -> float:
    # Convert dict to 2D array
    # Each row is a category, each column is a metric
    scores = np.array(list(perclass_score.values()))

    # Average of each column
    overall_score = np.mean(scores, axis=0)

    return overall_score


if __name__ == "__main__":
    query_results_path = "./Rorschach/evaluation/data/collect_neighbours_knn.csv"
    plot_type = "F1 score"
    column = "Apartment"
    k = None  # Top k results to consider

    TPs, FPs, TNs, FNs = calculate_perclass(query_results_path, plot_type, k)
    perclass_results = calculate_metrics(TPs, FPs, TNs, FNs)
    print(f"Perclass results for {column}: {perclass_results[column]}")

    overall_scores = calculate_overall(perclass_results)
    overall_scores = [round(score, 3) for score in overall_scores]
    results_columns = ["Precision", "Recall", "Accuracy", "Sensitivity", "Specificity", "F1 score"]
    print(f"Overall results: {dict(zip(results_columns, overall_scores))}")

    # Get F1 score of each class
    pc_f1 = {key: value[-1] for key, value in perclass_results.items()}
    plot_perclass_metrics(pc_f1, plot_type, k)
