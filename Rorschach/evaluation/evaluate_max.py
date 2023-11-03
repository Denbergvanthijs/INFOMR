import pandas as pd
import statistics as stats
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


def append_to_dict(_dict, label, value):
    if not (label in _dict.keys()):
        _dict[label] = [value]
    else:
        _dict[label].append(value)


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
    precisions = {}
    recalls = {}
    f1_scores = {}
    accuracies = {}
    sensitivities = {}
    specificities = {}
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

        # Compute performance metrics
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)

        if (precision + recall) == 0:
            f1_score = 0
        else:
            f1_score = 2 * ((precision * recall) / (precision + recall))

        # Store performance metric results
        for metric, value in zip([precisions, recalls, f1_scores, accuracies, sensitivities, specificities],
                                 [precision, recall, f1_score, accuracy, sensitivity, specificity]):
            append_to_dict(metric, query_class, value)

    # Aggregate performance metrics for each class
    perclass_precisions = {label: stats.mean(class_precisions) for label, class_precisions in precisions.items()}
    perclass_recalls = {label: stats.mean(class_recalls) for label, class_recalls in recalls.items()}
    perclass_f1_scores = {label: stats.mean(class_f1_scores) for label, class_f1_scores in f1_scores.items()}
    perclass_accuracies = {label: stats.mean(class_accuracies) for label, class_accuracies in accuracies.items()}
    perclass_sensitivities = {label: stats.mean(class_sensitivities) for label, class_sensitivities in sensitivities.items()}
    perclass_specificities = {label: stats.mean(class_specificities) for label, class_specificities in specificities.items()}

    return perclass_precisions, perclass_recalls, perclass_f1_scores, perclass_accuracies, perclass_sensitivities, perclass_specificities


def calculate_overall(perclass_precisions, perclass_recalls, perclass_f1_scores,
                      perclass_accuracies, perclass_sensitivities, perclass_specificities):
    # Aggregate performance metrics across entire database
    overall_precision = stats.mean(list(perclass_precisions.values()))
    overall_recall = stats.mean(list(perclass_recalls.values()))
    overall_f1_score = stats.mean(list(perclass_f1_scores.values()))
    overall_accuracy = stats.mean(list(perclass_accuracies.values()))
    overall_sensitivity = stats.mean(list(perclass_sensitivities.values()))
    overall_specificity = stats.mean(list(perclass_specificities.values()))

    print("\n" + "-"*30 + "\nOVERALL PERFORMANCE\n" + "-"*30)
    print("Overall precision: ", overall_precision)
    print("Overall recall: ", overall_recall)
    print("Overall F1 score: ", overall_f1_score)
    print("Overall accuracy: ", overall_accuracy)
    print("Overall sensitivity: ", overall_sensitivity)
    print("Overall specificity: ", overall_specificity)

    return overall_precision, overall_recall, overall_f1_score, overall_accuracy, overall_sensitivity, overall_specificity


if __name__ == "__main__":
    query_results_path = "./Rorschach/evaluation/data/collect_neighbours_knn.csv"
    plot_type = "F1 score"
    k = None  # Top k results to consider

    pc_p, pc_r, pc_f1, pc_acc, pc_sens, pc_spec = calculate_perclass(query_results_path, plot_type, k)

    overall_p, overall_r, overall_f1, overall_acc, overall_sens, overall_spec = calculate_overall(
        pc_p, pc_r, pc_f1, pc_acc, pc_sens, pc_spec)

    plot_perclass_metrics(pc_f1, plot_type, k)
