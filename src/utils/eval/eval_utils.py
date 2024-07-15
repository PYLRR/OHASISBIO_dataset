import numpy as np


def evaluate_peaks(ground_truth, detections, allowed_error):
    """ Compute the scores of segmenter detections using the peaks kept in each segment.
    Abreviations used : TP (True Positive), FP (False Positive), FN (False Negative), P (Positive), N (Negative)
    :param ground_truth: List giving, for each segment, the position of the events in s.
    :param detections: List giving, for each segment, the tuples (position of the detection in s,
    confidence value in [0,1]).
    :param allowed_error: Difference allowed between a detection and a ground truth event in s.
    :return: List of TP and number of FP when considering the whole segments (classification task),
    lists of TP and FP and number of N, P and FN when considering the peaks proximities (segmentation task).
    TP is expressed as a list of tuples (height, distance) where height is the maximum amplitude of the peak
    and distance the distance between the peak and the corresponding ground truth event.
    """
    # peaks prediction scores
    TP = []
    FP = 0
    # prediction scores of the whole segments rather than the peaks
    TP_per_seg = {}  # {seg_id : predicted_score}
    FP_per_seg = {}  # {seg_id : predicted_score}
    N_per_seg = 0
    P_per_seg = 0
    FN_per_seg = []  # list of seg_ids

    # browse segments
    for i in range(len(detections)):
        if len(ground_truth[i]) == 0:
            N_per_seg += 1
            if len(detections[i]) > 0:
                heights = [det[1] for det in detections[i]]
                FP_per_seg[i] = np.max(heights)
                FP += len(detections[i])
            continue
        # dict associating each ground truth peak index to its corresponding model output peak
        P_per_seg += 1
        success = {}

        # browse peaks in the segment
        for det in detections[i]:
            pos = det[0]
            height = det[1]
            distances = pos - ground_truth[i]
            min_idx = np.argmin(np.abs(distances))
            if np.abs(distances[min_idx]) < allowed_error:
                # check this ground truth peak has not already been detected
                if min_idx in success.keys():
                    FP += 1
                    # replace if greater peak
                    if height > TP[success[min_idx]][0]:
                        TP[success[min_idx]] = (height, distances[min_idx])
                else:
                    # add the success to the list
                    success[min_idx] = len(TP)
                    TP.append((height, distances[min_idx]))
            else:
                FP += 1

        # the FP/TP per segment only increases by 1
        if len(detections[i]) > 0:
            heights = [det[1] for det in detections[i]]
            TP_per_seg[i] = np.max(heights)
        else:
            FN_per_seg.append(i)

    TN_per_seg = N_per_seg - len(list(FP_per_seg.keys()))

    return TP, FP, TP_per_seg, TN_per_seg, FP_per_seg, FN_per_seg, P_per_seg, N_per_seg


def compute_ROC(TP, P, FP, N, thresh_delta=0.001):
    """ Compute the TP rate and FP rate of a detector given TP, P, FP and N values.
    :param TP: List of true positives confidence values.
    :param P: Number of positives (that is max length of TP).
    :param FP: List of false positives confidence values.
    :param N: Number of negatives (that is max length of FP).
    :param thresh_delta: Resolution in the confidence threshold variation to obtain the data.
    :return: List of (TP_rate, FP_rate) values obtained when varying the threshold, usable to plot a ROC curve.
    """
    TPr = []
    FPr = []
    for thresh in np.arange(0, 1+thresh_delta, thresh_delta):
        TPr.append(np.count_nonzero(TP > thresh) / P)
        FPr.append(np.count_nonzero(FP > thresh) / N)
    return TPr, FPr

def compute_residuals_histogram(allowed_d, TP_residuals):
    """ Compute the nb of TP distant of i seconds from the closest ground truth event.
    :param allowed_d: Discrete list of allowed residuals we want to consider for the histogram.
    :param TP_residuals: List of tuples (height, residual) of TP events, as real values.
    :return: A dict object associating keys from allowed_d to the corresponding counted number of TP.
    """
    step = abs(allowed_d[1] - allowed_d[0])
    TP_by_distance = {i: 0 for i in allowed_d}
    for p in TP_residuals:
        diff = np.abs(p[1] - allowed_d)
        idx = np.argmin(diff)
        # check the residual is admissible
        if diff[idx] <= step:
            TP_by_distance[allowed_d[idx]] += 1
    return {residual_s : v / len(TP_residuals) for residual_s, v in TP_by_distance.items()}
