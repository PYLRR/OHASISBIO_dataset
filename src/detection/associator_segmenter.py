import itertools
import datetime

import numpy as np
from scipy.signal import find_peaks
from skimage.transform import resize

MIN_P = 0.5
MIN_P_MATES = 0.2
NB = 3

TIME_RES = 0.5 # duration, in s, of a sample in the associator results
HALF_FOCUS_SIZE = 16  # window to compare for associator -> keep what was used for training
MAX_SHIFT = 128  # max allowed shift resulting from associator

def get_embedder_similarities(d1, m1, d2, m2, max_shift=MAX_SHIFT, half_focus=HALF_FOCUS_SIZE, min_p=MIN_P_MATES,
                              time_res=TIME_RES):
    # get data
    half1, half2 = half_focus, max_shift + half_focus
    full1, full2 = 2 * half1 + 1, 2 * half2 + 1
    dt1, dt2 = datetime.timedelta(seconds=half1 * time_res), datetime.timedelta(seconds=half2 * time_res)
    data1, data2 = m1.getSegment(d1 - dt1, d1 + dt1), m2.getSegment(d2 - dt2, d2 + dt2)
    data1, data2 = resize(data1, (16, full1)), resize(data2, (16, full2))

    # transform data to sliding array
    data2 = [data2[:, half2 + i - half_focus:half2 + i + half_focus + 1] for i in range(-max_shift, max_shift + 1)]
    data2 = np.array(data2)

    # get difference
    diff = np.sqrt(np.sum((data1[np.newaxis, :, :] - data2) ** 2, axis=(1, 2)))
    diff = diff / np.sqrt(full1 * 16 * 2)  # normalisation by max theoretical value
    if min_p is None:
        return diff

    # get peaks
    peaks, heights = find_peaks(-diff, height=-min_p, prominence=min_p / 8, width=5)
    heights = heights["peak_heights"]
    peaks_s = np.round((peaks - max_shift) * TIME_RES).astype(np.int32)
    if len(peaks_s) > 0:
        idx = np.argsort(np.abs(heights))
        peaks, height = peaks_s[idx], -heights[idx]
    else:
        peak, height = [0], [None]
    peaks = [datetime.timedelta(seconds=int(peak)) for peak in peaks]

    return peaks, height


def find_candidates(detection, d_h, tolerance, embedding_managers, min_p=MIN_P_MATES):
    d1, s1 = detection[0], detection[-1]
    if s1 not in d_h:
        return {}
    candidates = {}
    for s2 in d_h.keys():
        if s1 == s2:
            continue

        d2 = d1 + d_h[s1][s2]
        delta = tolerance
        max_delta = delta + datetime.timedelta(seconds=TIME_RES * (MAX_SHIFT + HALF_FOCUS_SIZE))
        if d2 + 2 * max_delta > embedding_managers[s2].dataset_end or d2 - 2 * max_delta < embedding_managers[
            s2].dataset_start or \
                d1 + max_delta > embedding_managers[s1].dataset_end or d1 - max_delta < embedding_managers[
            s1].dataset_start:
            continue
        p, h = get_embedder_similarities(d1, embedding_managers[s1], d2, embedding_managers[s2],
                                         max_shift=round(delta.total_seconds() / TIME_RES), min_p=min_p)
        if h[0] is None:
            # there was no acceptable peak
            continue
        candidates[s2] = []
        for i in range(len(p)):
            candidates[s2].append((d2 + p[i], h[i], s2))
        candidates[s2] = np.array(candidates[s2])
    return candidates


def best_matchup(scores, nb, d_h, tolerance):
    scores_merged = []
    for s in scores.keys():
        for j, score in enumerate(scores[s]):
            scores_merged.append((s, j, score[1]))

    scores_merged.sort(key=lambda x: x[2])
    selected = []
    chosen_idxs = set()
    idx = 0
    while len(selected) != nb and idx < len(scores_merged):
        s, j, similarity = scores_merged[idx]
        if s not in chosen_idxs:
            consistent = True
            d = scores[s][j][0]
            for (s2, j2) in selected:
                d2 = scores[s2][j2][0]
                if not (d + d_h[s][s2] - tolerance < d2 < d + d_h[s][s2] + tolerance):
                    consistent = False  # given location constraint, this can't be true
            if consistent:
                selected.append((s, j))
                chosen_idxs.add(s)
        idx += 1
    return selected


def best_matchups_combinatory(scores, nb, d_h, tolerance, to_keep=3, min_dist=1):
    scores_val = [np.array(scores[s])[:, 1] for s in scores.keys()]
    best_stations = np.argsort([np.min(sc) for sc in scores_val])
    # now remove stations that are close in space
    kept_stations = []
    for si in best_stations:
        s = list(scores.keys())[si]
        d = [np.sqrt(np.sum((np.array(list(scores.keys())[si2].get_pos()) - np.array(s.get_pos())) ** 2)) for si2 in
             kept_stations if si2 != si]
        if len(d) == 0 or np.min(d) > min_dist:
            kept_stations.append(si)
    kept_stations = kept_stations[:nb]
    kept_stations = np.array(list(scores.keys()))[kept_stations]
    if len(kept_stations) < nb:
        return []
    scores = {s: scores[s][:to_keep] for s in kept_stations}

    matchups = list(itertools.product(*list(scores.values())))
    matchup_scores = []
    to_remove = set()  # removable matchups because they violate d_h distances
    for i, matchup in enumerate(matchups):
        score = 0
        consistent = True
        for s1_idx, (d1, p1, s1) in enumerate(matchup):
            for s2_idx, (d2, p2, s2) in enumerate(matchup[s1_idx + 1:]):
                if not (d1 + d_h[s1][s2] - tolerance < d2 < d1 + d_h[s1][s2] + tolerance):
                    consistent = False  # given location constraints, this can't be true
            score += p1
        if consistent:
            matchup_scores.append(score)
        else:
            to_remove.add(i)
    matchups = np.delete(matchups, list(to_remove), axis=0)
    matchups = matchups[np.argsort(matchup_scores)]
    return matchups

def constrain_coord(coords):
    coords[0] = (coords[0] + 90) % 180 - 90
    coords[1] = (coords[1] + 180) % 360 - 180

    if np.isclose(coords[1], 180) or np.isclose(coords[1], -180) or np.isclose(coords[0], 90) or np.isclose(coords[0], -90):
        return [None, None]

    return coords

def locate(matchup, sound_model, cost_allowed=None, initial_pos=None):
    det_times = [c[0] for c in matchup]
    det_pos = [c[-1].get_pos() for c in matchup]
    try:
        r = sound_model.localize_common_source(det_pos, det_times, initial_pos=initial_pos)
    except AssertionError:
        return False, None
    if type(r) != list and np.count_nonzero(np.isnan(r.x))==0:
        r.x[1:] = constrain_coord(r.x[1:])
        if r.x[1] is not None and (not cost_allowed or np.sqrt(r.cost/len(det_times)) < cost_allowed):
            return True, r
        else:
            return False, r
    return False, None