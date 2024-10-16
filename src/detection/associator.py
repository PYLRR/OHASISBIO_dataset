import copy
from pathlib import Path
import datetime

import numpy as np
from tqdm import tqdm
from line_profiler_pycharm import profile

from utils.data_reading.sound_data.station import StationsCatalog
from utils.physics.sound.sound_model import HomogeneousSoundModel
from utils.physics.sound.sound_velocity_grid import MonthlySoundVelocityGridOptimized

import time


def best_matchup(scores, nb, to_skip=0):
    scores_merged = []
    for i in range(len(scores)):
        for j, s in enumerate(scores[i]):
            scores_merged.append((i, j, s))

    scores_merged.sort(key=lambda x: x[2])
    selected = []
    chosen_idxs = set()
    idx = 0
    n_skipped = 0
    while len(selected) != nb:
        i, j, s = scores_merged[idx]
        if i not in chosen_idxs:
            if len(selected) == nb - 1:
                if n_skipped < to_skip:
                    n_skipped += 1
                    continue
            selected.append((i, j))
            chosen_idxs.add(i)
        idx += 1
    return selected


def constrain_coord(coords):
    coords[0] = (coords[0] + 90) % 180 - 90
    if np.isclose(coords[0], 90):
        coords[0] = 89.75
    if np.isclose(coords[0], -90):
        coords[0] = -89.75

    coords[1] = (coords[1] + 180) % 360 - 180
    if np.isclose(coords[1], 180):
        coords[1] = 179.75
    if np.isclose(coords[1], -180):
        coords[1] = -179.75

    return coords


def load_processed(res_file):
    if not Path(res_file).exists():
        return None, set()
    with open(res_file, "r") as f:
        lines = f.readlines()
    last_seen = [datetime.datetime.strptime(l.split(",")[0], "%Y%m%d_%H%M%S") for l in lines][-1]
    seen_dates = set()
    for line in lines:
        dates = line.split(",")[7::2]
        for date in dates:
            date = date.strip()
            if len(date) == 0:
                continue
            seen_dates.add(datetime.datetime.strptime(date, "%Y%m%d_%H%M%S"))
    return last_seen, seen_dates


def find_candidates(detections, date, s1, embedding, d_h, seen_dates, idx_detections, stations):
    candidates = []
    for s2 in stations:
        candidates.append([])
        detections_s2 = detections[s2]
        while idx_detections[s2] > 0 and (
                idx_detections[s2] == len(detections[s2]) or detections_s2[idx_detections[s2]][0] > date - d_h[s1][s2]):
            idx_detections[s2] -= 1
        while idx_detections[s2] < len(detections[s2]) and detections_s2[idx_detections[s2]][0] < date + d_h[s1][s2]:
            det = detections_s2[idx_detections[s2]]
            if det[0] > date - d_h[s1][s2]:
                delta_embedding = np.sqrt(np.sum(np.square(det[2] - embedding))) / np.sqrt(16)
                if det[0] not in seen_dates:
                    candidates[-1].append((det[0], det[1], delta_embedding, s2))
            idx_detections[s2] += 1
        if len(candidates[-1]) == 0:
            candidates = candidates[:-1]
        else:
            candidates[-1] = np.array(candidates[-1])
    return candidates


def locate(matchup, sound_model, cost_allowed=None, initial_pos=None):
    det_times = [c[0] for c in matchup]
    det_pos = [c[-1].get_pos() for c in matchup]
    try:
        r = sound_model.localize_common_source(det_pos, det_times, initial_pos=initial_pos)
        if type(r) != list and (not cost_allowed or r.cost < cost_allowed):
            r.x[1:] = constrain_coord(r.x[1:])
            return r
    except:
        pass  # we return None
    return None

if __name__=="__main__":
    # load data
    datasets_yaml = "/home/plerolland/Bureau/dataset.yaml"
    sound_model_h = HomogeneousSoundModel()
    sound_model_g = MonthlySoundVelocityGridOptimized(
        [f"../../data/sound_model/min-velocities_month-{i:02d}.nc" for i in range(1, 13)], interpolate=True)
    stations_c = StationsCatalog(datasets_yaml).filter_out_undated().filter_out_unlocated()
    year = 2018
    RES_FILE = f"../../data/detections/{year}/loc.csv"
    detections = np.load(f"../../data/detections/{year}/detections.npy", allow_pickle=True).item()
    MIN_P = 0.4
    NB = 3
    print(f"detections loaded")

    # get inter-station distances
    stations = list(detections.keys())
    d_h = {s1: {s2: datetime.timedelta(seconds=0) for s2 in stations} for s1 in stations}
    for s1 in stations:
        for s2 in stations:
            d_h[s1][s2] = datetime.timedelta(
                seconds=sound_model_h.get_sound_travel_time(s1.get_pos(), s2.get_pos()))
    print(f"inter-station distances computed")

    # merge all detections, sort them by date, filter them by probability and prepare for browsing
    merged_detections = []
    for s, dets in detections.items():
        for det in dets:
            merged_detections.append((det[0], det[1], tuple(det[2]), s))
    merged_detections = np.array(merged_detections, dtype=np.object_)
    merged_detections = merged_detections[np.argsort(merged_detections[:, 0])]
    to_browse = merged_detections[merged_detections[:, 1] > MIN_P]
    idx_detections = {s: 0 for s in stations}
    last_seen, seen_dates = load_processed(RES_FILE)
    print(f"to-browse list built")

    # browsing
    for (date, p, embedding, s1) in tqdm(to_browse):
        if date in seen_dates or last_seen is not None and date < last_seen:
            continue

        # initialize candidates
        candidates = find_candidates(detections, date, s1, embedding, d_h, seen_dates, idx_detections, stations)
        if len(candidates) < NB:
            continue

        # perform first (and simplest) location attempt
        scores = [c[:, 2] for c in candidates]
        best_m = best_matchup(scores, nb=NB)
        matchup = [(date, p, embedding, s1)] + [candidates[i][j] for i, j in best_m]
        loc_res = locate(matchup, sound_model_h, 500)

        # in case it worked, go further
        if loc_res:
            loc_res = locate(matchup, sound_model_g, 300, initial_pos=loc_res.x[1:])
            if loc_res:
                det_times = [date] + [c[0] for c in matchup]
                date_event = np.min(det_times) + datetime.timedelta(seconds=loc_res.x[0])

                # add other stations
                changed = False
                loc_res_updated = copy.deepcopy(loc_res)
                matchup_updated = copy.deepcopy(matchup)

                seen_stations = set([c[-1] for c in matchup])
                for c in candidates:
                    if c[0][-1] in seen_stations:
                        continue
                    expected_time = date_event + datetime.timedelta(
                        seconds=sound_model_h.get_sound_travel_time(c[0][-1].get_pos(), loc_res_updated.x[1:]))
                    found = False
                    for chosen in c:
                        if np.abs(chosen[0] - expected_time).total_seconds() < 10:
                            found = True
                            break
                    if not found:
                        continue
                    matchup_updated = matchup_updated + [chosen]
                    loc_res_new = locate(matchup_updated, sound_model_h, 500)
                    if loc_res_new:  # update location
                        changed = True
                        seen_stations.add(chosen[-1])
                        loc_res_updated = loc_res_new
                    else:  # rollback
                        matchup_updated = matchup_updated[:-1]

                if changed:
                    loc_res_new = locate(matchup_updated, sound_model_g, 300, initial_pos=loc_res_updated.x[1:])
                    if loc_res_new:
                        loc_res = loc_res_new
                        matchup = matchup_updated
                if loc_res:
                    det_times = [date] + [c[0] for c in matchup]
                    for d in det_times:
                        seen_dates.add(d)  # to avoid using the same detection twice
                    date_event = np.min(det_times) + datetime.timedelta(seconds=loc_res.x[0])
                    try:
                        J = loc_res.jac
                        cov = np.linalg.inv(J.T.dot(J))
                        var = np.sqrt(np.diagonal(cov))
                    except:
                        var = [-1, -1, -1]

                    to_write = f'{date_event.strftime("%Y%m%d_%H%M%S")},{loc_res.x[1]:.4f},{loc_res.x[2]:.4f},{var[0]:.4f},{var[1]:.4f},{var[2]:.4f}'
                    for d, _, _, s in matchup:
                        to_write += f',{s.name}-{s.date_start.year},{d.strftime("%Y%m%d_%H%M%S")}'
                    with open(RES_FILE, "a") as f:
                        f.write(to_write + "\n")


# TODO gradually constrain inter-stations distances