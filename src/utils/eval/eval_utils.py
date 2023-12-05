import numpy as np


# compute the scores of detection
# ground_truth : list giving, for each segment, the position of the events in s
# detections : list giving, for each segment, the tuples (position of the detection in s, confidence value in [0,1])
# allowed_error : difference allowed between a detection and a ground truth event in s
def evaluate_peaks(ground_truth, detections, allowed_error):
    # peaks prediction scores
    TP = []
    FP = 0
    # prediction scores of the whole segments rather than the peaks
    TP_per_seg = []
    FP_per_seg = []
    N_per_seg = 0
    P_per_seg = 0
    FN_per_seg = 0

    # browse segments
    for i in range(len(detections)):
        if len(ground_truth[i]) == 0:
            N_per_seg += 1
            if len(detections[i]) > 0:
                heights = [det[1] for det in detections[i]]
                FP_per_seg.append(np.max(heights))
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
            TP_per_seg.append(np.max(heights))
        else:
            FN_per_seg += 1

    TN_per_seg = N_per_seg - len(FP_per_seg)
    TP_per_seg = np.array(TP_per_seg)

    return TP, FP, TP_per_seg, TN_per_seg, FP_per_seg, FN_per_seg, P_per_seg, N_per_seg


# compute the TP rate and FP rate of a detector
# thresh_delta : resolution in the threshold variation to obtain the data
# TP : list of true positives confidence values
# P : number of positives (that is max length of TP)
# FP : list of false positives confidence values
# N : number of negatives (that is max length of FP)
def compute_ROC(TP, P, FP, N, thresh_delta=0.001):
    TPr = []
    FPr = []
    for thresh in np.arange(0, 1, thresh_delta):
        TPr.append(np.count_nonzero(TP > thresh) / P)
        FPr.append(np.count_nonzero(FP > thresh) / N)
    return TPr, FPr
