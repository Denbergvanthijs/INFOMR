from collections import defaultdict

import numpy as np
import pandas as pd
from plot_results import plot_perclass_metrics
from sklearn.metrics import confusion_matrix
from tqdm.contrib import tzip


def get_query_results(results_path: str) -> tuple:
    """Gets all pairs of query category and match categories from the results csv file.

    :param results_path: Path to the csv file containing the results of the query
    :type results_path: str
    :return: List of nd.arrays containing the categories of the matches for each query, and a list of the ground truths
    :rtype: tuple
    """
    # columns: query_filepath,query_category,match_filepath,match_category,distance
    df = pd.read_csv(results_path, sep=",")

    # For each query, get a list of the matched categories
    query_results = []
    ground_truths = []
    for query_filepath in df["query_filepath"].unique():
        # Get all rows that match the current query
        df_query = df[df["query_filepath"] == query_filepath]
        # The categories of all matches for the current query
        all_matches = df_query["match_category"].tolist()
        # The category of the current query, select single value from df
        gts_match = df_query["query_category"].unique()

        # Check that there is only a single ground truth for the query
        if gts_match.shape[0] > 1:
            raise ValueError(f"Multiple ground truths for query {query_filepath}")

        # Save to lists, order is important
        query_results.append(np.array(all_matches))
        ground_truths.append(gts_match[0])  # Select single value from array

    return query_results, np.array(ground_truths)


def calculate_perclass(query_results_path: str, plot_type: str, k: int = None):
    query_results, ground_truths = get_query_results(query_results_path)

    # Compute metrics
    TPs = defaultdict(list)
    FPs = defaultdict(list)
    TNs = defaultdict(list)
    FNs = defaultdict(list)

    database_size = len(query_results)  # Number of shapes in database

    for y_pred, query_class in zip(query_results, ground_truths):
        if k is not None:  # Limit to the top k results, results are already sorted by distance
            y_pred = y_pred[:k]

        # Number of meshes retrieved, should ideally be equal to k
        # But can be less if k > category size or if there are not enough matches in the database
        query_size = y_pred.shape[0]

        not_retrieved_size = database_size - query_size  # Number of shapes not retrieved
        not_retrieved_data = np.zeros(not_retrieved_size)  # Create array of zeros for shapes not retrieved

        # 1 for all retrieved shapes, 0 for all shapes not retrieved
        # Thus, the shapes at (database size minus k) are always 0
        y_true = np.concatenate((np.ones(query_size), not_retrieved_data))

        # y_retrieved should be same length as y_true
        y_retrieved = np.where(y_pred == query_class, 1, 0).astype(int)  # Number of retrieved shapes with either 0 or 1: TP + FP
        y_missing_ones = np.ones(query_size - y_retrieved.sum(), dtype=int)  # Number of relevant shapes that are not retrieved: FN
        y_zeros = np.zeros(not_retrieved_size - y_missing_ones.sum(), dtype=int)  # Number of shapes not retrieved that should be 0: TN
        y_non_retrieved = np.concatenate((y_missing_ones, y_zeros))  # Non-retrieved shapes are either TN or FN

        y_pred = np.concatenate((y_retrieved, y_non_retrieved))

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])  # Set labels as 0 and 1 to guarantee correct order
        TN, FP, FN, TP = cm.ravel()

        # Store performance metric results
        for metric, value in zip([TPs, FPs, TNs, FNs], [TP, FP, TN, FN]):
            metric[query_class].append(value)

    return TPs, FPs, TNs, FNs


def f_beta_score(precision, recall, beta):
    return (1 + beta**2) * ((precision * recall) / ((beta**2 * precision) + recall))


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
        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)
        f1_score = f_beta_score(precision, recall, 1)
        f2_score = f_beta_score(precision, recall, 2)

        results = [precision, recall, accuracy, sensitivity, specificity, f1_score, f2_score]

        results = np.nan_to_num(results, nan=0, neginf=0, posinf=0)  # Replace NaNs with 0
        results = np.mean(results, axis=1)  # Average over all queried shapes of a category

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
    query_results_paths = ["./Rorschach/evaluation/data/collect_neighbours_knn.csv",
                           "./Rorschach/evaluation/data/collect_neighbours_manhattan.csv",
                           "./Rorschach/evaluation/data/collect_neighbours_euclidean.csv",
                           "./Rorschach/evaluation/data/collect_neighbours_cosine.csv",
                           "./Rorschach/evaluation/data/collect_neighbours_emd.csv"]
    distance_functions = ["KNN", "Manhattan", "Euclidean", "Cosine", "EMD"]
    results_columns = ["Precision", "Recall", "Accuracy", "Sensitivity", "Specificity", "F1 score", "F2 score"]
    plot_type = "F2 score"
    column = "Apartment"
    ks = [3, None]  # Top k results to consider

    results = []
    for fp, distance_function in tzip(query_results_paths, distance_functions):
        for k in ks:
            TPs, FPs, TNs, FNs = calculate_perclass(fp, plot_type, k)
            perclass_results = calculate_metrics(TPs, FPs, TNs, FNs)
            # print(f"Perclass results for {column}: {perclass_results[column]}")

            overall_scores = calculate_overall(perclass_results)
            overall_scores = [round(score, 3) for score in overall_scores]
            # print(f"Overall results: {dict(zip(results_columns, overall_scores))}")

            # Get F1 score of each class
            pc_f1 = {key: value[-1] for key, value in perclass_results.items()}
            plot_perclass_metrics(pc_f1, plot_type, distance_function, k)

            k = k if k is not None else "all"  # Nonetype will not save correctly to csv
            results.append([distance_function, k] + overall_scores)

    # Print results
    results = pd.DataFrame(results, columns=["Distance function", "k"] + results_columns)
    print(results)
    results.to_csv("./Rorschach/evaluation/data/overall_results.csv", index=False, sep=",")
