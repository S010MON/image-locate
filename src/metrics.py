import numpy as np


def recall_at_k(distances: np.ndarray, testing=False) -> tuple:
    """
    Calculate the recall @k for top 1, 5, 10 and 1%, 5%, 10%
    :param distances: a distance matrix of shape (gnd_size, sat_size)
    :return: a tuple of (top_1, top_5, top_10, top_percent) of the images recalled over the whole dataset
    """
    count = distances.shape[0]
    correct_dists = distances.diagonal()
    sorted_dists = np.sort(distances, axis=1)

    top_1 = np.sum(correct_dists <= sorted_dists[:, 1]) / count * 100
    top_5 = np.sum(correct_dists <= sorted_dists[:, 5]) / count * 100
    top_10 = np.sum(correct_dists <= sorted_dists[:, 10]) / count * 100
    if not testing:
        top_50 = np.sum(correct_dists <= sorted_dists[:, 50]) / count * 100
        top_100 = np.sum(correct_dists <= sorted_dists[:, 100]) / count * 100
        top_500 = np.sum(correct_dists <= sorted_dists[:, 500]) / count * 100

    one_percent_idx = int(float(count) * 0.01)
    top_one_percent = np.sum(correct_dists <= sorted_dists[:, one_percent_idx]) / count * 100

    five_percent_idx = int(float(count) * 0.05)
    top_five_percent = (np.sum(correct_dists <= sorted_dists[:, five_percent_idx])) / count * 100

    ten_percent_idx = int(float(count) * 0.1)
    top_ten_percent = (np.sum(correct_dists <= sorted_dists[:, ten_percent_idx])) / count * 100

    print("1% idx ", one_percent_idx, "| 5% idx ", five_percent_idx, "| 10% idx ", ten_percent_idx)
    print(f"top: 1={sorted_dists[0, 1]}, 5={sorted_dists[0, 5]} 10={sorted_dists[0, 10]}\n"
          f"top: 1%={sorted_dists[0, one_percent_idx]}, 5%={sorted_dists[0, five_percent_idx]} 10%={sorted_dists[0, ten_percent_idx]}")

    if testing:
        return top_1, top_5, top_10, top_one_percent, top_five_percent, top_ten_percent

    return top_1, top_5, top_10, top_50, top_100, top_500, top_one_percent, top_five_percent, top_ten_percent


def proportional_search_space_reduction(distances: np.ndarray) -> tuple:
    """
    Calculates the proportional reduction in the search space if the correct image is
    retained in the dataset, and all further images are discarded.
    :param distances: a distance matrix of shape (gnd_size, sat_size)
    :return: a tuple consisting of:
        (mean_pssr,
         max_pssr,
         min_pssr,
         mean_boundary,
         max_boundary,
         min_boundary)
    """
    boundaries = distances.diagonal()
    count_total = len(distances)
    count_greater = np.sum(np.greater_equal(boundaries, distances), axis=0)
    pssr = (count_greater / count_total)
    return np.mean(pssr), pssr[np.argmax(pssr)], pssr[np.argmin(pssr)], \
        np.mean(boundaries), boundaries[np.argmax(boundaries)], boundaries[np.argmin(boundaries)]
