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


def plot_perclass_metrics(data_dict, metric):
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
    plt.show()


def main(query_results_path):
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
        query_size = len(y_pred)

        # Get true positives/negatives and false positives/negatives
        TP = y_pred.count(query_class)          # Correctly labelled as 'member of query class'
        # Incorrectly labelled as 'member of query class' (i.e. all returned shapes that are not part of the query class)
        FP = query_size - TP
        # Correctly labelled as 'NOT a member of query class' (i.e. all shapes in the database not part of query class that were not returned)
        TN = database_size - query_size - FP
        # Incorrectly labelled as 'NOT a member of query class' (i.e. all shapes in the database that are
        FN = query_size - TP
        # a part of the query class but were not returned)

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
        append_to_dict(precisions, query_class, precision)
        append_to_dict(recalls, query_class, recall)
        append_to_dict(f1_scores, query_class, f1_score)
        append_to_dict(accuracies, query_class, accuracy)
        append_to_dict(sensitivities, query_class, sensitivity)
        append_to_dict(specificities, query_class, specificity)

    # Aggregate performance metrics for each class
    perclass_precisions = {label: stats.mean(class_precisions) for label, class_precisions in precisions.items()}
    perclass_recalls = {label: stats.mean(class_recalls) for label, class_recalls in recalls.items()}
    perclass_f1_scores = {label: stats.mean(class_f1_scores) for label, class_f1_scores in f1_scores.items()}
    perclass_accuracies = {label: stats.mean(class_accuracies) for label, class_accuracies in accuracies.items()}
    perclass_sensitivities = {label: stats.mean(class_sensitivities) for label, class_sensitivities in sensitivities.items()}
    perclass_specificities = {label: stats.mean(class_specificities) for label, class_specificities in specificities.items()}
    # print("\nper-class mean precisions: ", perclass_precisions)
    # print("\nper-class mean recalls: ", perclass_recalls)
    # print("\nper-class mean f1 scores: ", perclass_f1_scores)
    # print("\nper-class mean accuracies: ", perclass_accuracies)
    # print("\nper-class mean sensitivities: ", perclass_sensitivities)
    # print("\nper-class mean specificities: ", perclass_specificities)

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

    # Plot per-class performance metrics
    plot_type = "accuracy"

    plot_d = {"precision": perclass_precisions,
              "recall": perclass_recalls,
              "F1 score": perclass_f1_scores,
              "accuracy": perclass_accuracies,
              "sensitivity": perclass_sensitivities,
              "specificity": perclass_specificities}

    plot_perclass_metrics(plot_d[plot_type], plot_type)


if __name__ == "__main__":
    query_results_path = "./Rorschach/evaluation/data/collect_neighbours_knn.csv"

    main(query_results_path)
