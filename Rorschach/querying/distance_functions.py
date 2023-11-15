import numpy as np


# Compute Manhatten distance between two vectors
def get_manhattan_distance(vec_a, vec_b, range_min: int = None, range_max: int = None, normalize=False):
    vec_a = np.array(vec_a)
    vec_b = np.array(vec_b)

    dist = np.abs(vec_a - vec_b).sum()

    if normalize:
        max_dist = (range_max - range_min) * len(vec_a)
        dist /= max_dist

    return dist


# Compute Euclidian distance between two vectors
def get_euclidean_distance(vec_a, vec_b, range_min: int = None, range_max: int = None, normalize=False):
    vec_a = np.array(vec_a)
    vec_b = np.array(vec_b)

    dist = np.linalg.norm(vec_a - vec_b)

    if normalize:
        max_dist = np.sqrt(len(vec_a) * ((range_max - range_min)**2))
        dist /= max_dist

    return dist


# Compute cosine dissimilarity between two vectors
def get_cosine_distance(vec_a, vec_b, normalize=False):
    cosine_similarity = np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    dist = 1 - cosine_similarity

    if normalize:
        dist /= 2

    return dist


# Compute Earth Mover's distance (EMD) between two feature vectors (only use if the minimal possible value of a feature is positive (>= 0))
def get_emd(features_1, features_2):
    i, j = 0, 1
    flow = [[0 for _ in range(len(features_1))] for _ in range(len(features_1))]
    difference = [0] * len(features_1)
    row = [0] * len(features_1)

    # Initialize empty flow matrix
    for p in range(len(features_1)):
        flow[p] = row.copy()
        flow[p][p] = min(features_1[p], features_2[p])
        difference[p] = features_1[p] - features_2[p]

    # Fill out the flow matrix by spreading differences
    while i + j < 2 * (len(features_1) - 1):
        if difference[i] > 0 and difference[j] < 0:
            if difference[i] <= -difference[j]:
                flow[j][i] = difference[i]
                difference[j] += difference[i]
                difference[i] = 0
                i += 1
                j = i + 1
            else:
                flow[j][i] = -difference[j]
                difference[i] += difference[j]
                difference[j] = 0
                if j < (len(features_1) - 1):
                    j += 1
                else:
                    i += 1
                    j = i + 1
        elif difference[i] < 0 and difference[j] > 0:
            if -difference[i] < difference[j]:
                flow[j][i] = -difference[i]
                difference[j] += difference[i]
                difference[i] = 0
                i += 1
                j = i + 1
            else:
                flow[i][j] = difference[j]
                difference[i] += difference[j]
                difference[j] = 0
                if j < len(features_1) - 1:
                    j += 1
                else:
                    i += 1
                    j = i + 1
        elif difference[i] == difference[j]:
            i += 1
            j = i + 1
        else:
            if j < len(features_1) - 1:
                j += 1
            else:
                i += 1
                j = i + 1

    # Compute sum of distance times flow
    work = 0
    for p in range(len(features_1)):
        for q in range(len(features_1)):
            work += abs(p - q) * flow[p][q]

    # 'Normalize' by dividing by total flow
    total_flow = 0
    for i in range(len(features_1)):
        for j in range(len(features_1)):
            total_flow += flow[i][j]
    emd = work / total_flow

    return emd


def zero_distance(features_1, features_2):
    return 0
