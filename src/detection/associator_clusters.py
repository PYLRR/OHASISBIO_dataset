import copy
import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

from scipy.stats import t as student
from associator import locate
from utils.data_reading.catalogs.ISC import ISC_file
from utils.data_reading.sound_data.station import StationsCatalog
from utils.physics.bathymetry.bathymetry_grid import BathymetryGrid
from utils.physics.sound.sound_model import HomogeneousSoundModel
from utils.physics.sound.sound_velocity_grid import MonthlySoundVelocityGridOptimized


def get_valid(allowed_delta, delta, IDs):
    valid = {ID: {} for ID in IDs}
    for ID in tqdm(IDs, position=0, leave=True):
        for ID_ in IDs:
            if (ID_ in delta[ID] and delta[ID][ID_] < allowed_delta) or ID == ID_:
                valid[ID][ID_] = True
    return valid

def add_to_cluster(current_clusters, ID_to_cluster, ID, cluster_idx):
    current_clusters[cluster_idx].append(ID)
    ID_to_cluster[ID] = cluster_idx

def merge_clusters(current_clusters, ID_to_cluster, cluster_1_idx, cluster_2_idx):
    for ID in current_clusters[cluster_2_idx]:
        add_to_cluster(current_clusters, ID_to_cluster, ID, cluster_1_idx)
    del current_clusters[cluster_2_idx]

def get_clusters(IDs, valid):
    clusters = {}
    ID_to_cluster = {}
    for ID in tqdm(IDs, position=0, leave=True):
        if ID not in ID_to_cluster:
            new_cluster_key = np.max(list(clusters.keys()) + [0]) + 1  # + [0] to handle empty list case
            clusters[new_cluster_key] = []
            add_to_cluster(clusters, ID_to_cluster, ID, new_cluster_key)

        for ID_ in valid[ID].keys():
            if ID_ in ID_to_cluster:
                if ID in ID_to_cluster and ID_to_cluster[ID] != ID_to_cluster[ID_]:
                    merge_clusters(clusters, ID_to_cluster, ID_to_cluster[ID], ID_to_cluster[ID_])
            else:
                add_to_cluster(clusters, ID_to_cluster, ID_, ID_to_cluster[ID])
    return clusters

def find_candidates(detections, detection, d_h, seen_dates, tolerance, min_p=0):
    date, embedding, s1 = detection[0], detection[2], detection[-1]
    if date in seen_dates:
        return []
    candidates = {}
    for s2 in detections.keys():
        if s1 == s2:
            continue
        detections_s2 = detections[s2]
        idx_min = np.searchsorted(detections[s2][:,0], date + d_h[s1][s2] - tolerance, side='right')
        idx = idx_min  # note : iterating like this instead of finding max with searchsorted is way more efficent
        while idx < len(detections[s2]) and detections[s2][idx,0] < date + d_h[s1][s2] + tolerance:
            det = detections_s2[idx]
            if det[0] not in seen_dates and det[1]>min_p:
                delta_embedding = np.sqrt(np.sum(np.square(det[2] - embedding))) / np.sqrt(16)
                if det[0] not in seen_dates:
                    candidates.setdefault(s2, []).append((det[0], det[1], delta_embedding, s2))
            idx += 1
        if s2 in candidates:
            candidates[s2] = np.array(candidates[s2])
    return candidates

def best_matchup(scores, nb, d_h, tolerance):
    scores_merged = []
    for s in scores.keys():
        for j, score in enumerate(scores[s]):
            scores_merged.append((s, j, score[2]))

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

def load_processed(res_file):
    if not Path(res_file).exists():
        return set(), set()
    with open(res_file, "r") as f:
        lines = f.readlines()
    seen_IDs = set()
    seen_dates = set()
    for line in lines:
        seen_IDs.add(int(line.split(",")[6]))
        dates = line.split(",")[9::2]
        for date in dates:
            date = date.strip()
            if len(date) == 0:
                continue
            seen_dates.add(datetime.datetime.strptime(date, "%Y%m%d_%H%M%S"))
    return seen_IDs, seen_dates

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
        if type(r) != list and np.count_nonzero(np.isnan(r.x))==0:
            r.x[1:] = constrain_coord(r.x[1:])
            if r.x[1] is not None and (not cost_allowed or np.sqrt(r.cost/len(det_times)) < cost_allowed):
                return True, r
            else:
                return False, r
    except:
        pass  # we return None
    return False, None


if __name__ == "__main__":
    # load data
    datasets_yaml = "/home/plerolland/Bureau/dataset.yaml"
    sound_model_h = HomogeneousSoundModel()
    sound_model_g = MonthlySoundVelocityGridOptimized(
        [f"../../data/sound_model/min-velocities_month-{i:02d}.nc" for i in range(1, 13)], interpolate=True)
    bathy_model = BathymetryGrid.create_from_NetCDF("../../data/geo/GEBCO_2023_sub_ice_topo.nc", lat_bounds=[-75, 35],
                                                    lon_bounds=[-20, 180])
    stations_c = StationsCatalog(datasets_yaml).filter_out_undated().filter_out_unlocated()
    year = 2018
    detections = np.load(f"../../data/detections/{year}/detections.npy", allow_pickle=True).item()
    isc_file = f"/home/plerolland/Bureau/catalogs/ISC/eqk_isc_{year}.txt"
    isc = ISC_file(isc_file)
    MIN_P = 0.4
    MIN_P_MATES = 0.2
    NB = 3
    TOLERANCE = datetime.timedelta(seconds=20)
    ALLOWED_DELTA = 30  # for ISC clustering
    TIME_DELTA_SEARCH = datetime.timedelta(seconds=5*86400)
    RES_FILE = f"../../data/detections/{year}/matchups_clusters_loc.csv"
    print(f"detections loaded")

    # filter isc events
    to_del = set()
    for ID, event in isc.items.items():
        if bathy_model.get_nearest_values(event.get_pos()) > 0:
            to_del.add(ID)
        if isc[ID].get_pos()[0] > -5 and isc[ID].get_pos()[1] > 115 or isc[ID].get_pos()[0] > -30 and isc[ID].get_pos()[
            1] > 130 or isc[ID].get_pos()[0] > -45 and isc[ID].get_pos()[1] > 170 \
                or isc[ID].get_pos()[0] > -20 and isc[ID].get_pos()[1] > 85:
            to_del.add(ID)
    for ID in to_del:
        del isc.items[ID]
    print(f"{len(to_del)} terrestrial events removed from catalog")



    # merge all detections and sort them by date
    stations = list(detections.keys())
    merged_detections = []
    for s, dets in detections.items():
        for det in dets:
            merged_detections.append((det[0], det[1], tuple(det[2]), s))
    merged_detections = np.array(merged_detections, dtype=np.object_)
    merged_detections = merged_detections[np.argsort(merged_detections[:, 0])]
    merged_detections_kept = merged_detections[merged_detections[:, 1] > MIN_P]
    print(f"to-browse list built")

    # compute deltas to clusterize ISC
    idx_last_valid = 0
    delta_km = {}
    delta_d = {}
    delta = {}
    IDs = list(isc.items.keys())
    for i, (ID, isc_event) in tqdm(enumerate(isc.items.items()), total=len(IDs), position=0, leave=True):
        if i == 0:
            continue
        date, pos = isc_event.date, isc_event.get_pos()

        # peremption of clusters (by date)
        while idx_last_valid < i and isc[IDs[idx_last_valid]].date < date - datetime.timedelta(days=ALLOWED_DELTA):
            idx_last_valid += 1

        dates, poss = [], []
        for idx in range(idx_last_valid, i):
            poss.append(isc[IDs[idx]].get_pos())
            dates.append(isc[IDs[idx]].date)
        dates, poss = np.array(dates), np.array(poss)
        # approximation of 1) loxodromy and 2) constant deg to km conversion ; acceptable because small distances and "far" from pole
        deltas_km = np.sqrt((poss[:, 0] - pos[0]) ** 2 + (poss[:, 1] - pos[1]) ** 2) * 111
        deltas_d = np.array([d.total_seconds() / 86400 for d in np.abs(dates - date)])
        deltas = np.sqrt(deltas_km ** 2 + deltas_d ** 2)

        for idx in range(idx_last_valid, i):
            ID_ = IDs[idx]
            delta_km.setdefault(ID, {})[ID_] = deltas_km[idx - idx_last_valid]
            delta_d.setdefault(ID, {})[ID_] = deltas_d[idx - idx_last_valid]
            delta.setdefault(ID, {})[ID_] = deltas[idx - idx_last_valid]

    # make it symmetric, expected run time ~ 40 s for above conditions
    for ID in IDs:
        for ID_ in delta.setdefault(ID, {}).keys():
            delta_km.setdefault(ID_, {})[ID] = delta_km[ID][ID_]
            delta_d.setdefault(ID_, {})[ID] = delta_d[ID][ID_]
            delta.setdefault(ID_, {})[ID] = delta[ID][ID_]

    valid = get_valid(ALLOWED_DELTA, delta, IDs)
    clusters = get_clusters(IDs, valid)
    date_min = [np.min([isc[ID].date for ID in cluster]) for cluster in clusters.values()]
    cluster_centroids = [np.mean([isc[ID].get_pos() for ID in cluster], axis=0) for cluster in clusters.values()]
    clusters = list(clusters.values())
    clusters = [clusters[i] for i in np.argsort(date_min)]
    print(f"clusters computed, found {len(clusters)} out of {len(IDs)} events")

    seen_IDs, seen_dates = load_processed(RES_FILE)

    # now compute matchups
    print("starting matchup computation")
    for i, cluster in enumerate(clusters):
        if cluster[0] in seen_IDs:
            continue
        if len(cluster) > 0:  # generalize to all clusters including singletons
            centroid = np.mean([isc[ID].get_pos() for ID in cluster], axis=0)
            dates = [isc[ID].date for ID in cluster]
            date_min, date_max = np.min(dates)-TIME_DELTA_SEARCH, np.max(dates)+TIME_DELTA_SEARCH
            date_mid = date_min + (date_max - date_min) / 2

            expected = {s: sound_model_h.get_sound_travel_time(centroid, s.get_pos(), date=date_mid) for s in stations}
            # d_h[s1][s2] = given a detection on s1, time to wait before getting it on s2 (can be negative)
            d_h = {s1: {s2: datetime.timedelta(seconds=expected[s2] - expected[s1]) for s2 in stations} for s1 in
                   stations}

            # in bounds search! (inter-search)
            idx_detections = {s: 0 for s in stations}
            idx_min = np.searchsorted(merged_detections_kept[:, 0], date_min, side='right')
            idx_max = np.searchsorted(merged_detections_kept[:, 0], date_max, side='left')
            for idx in tqdm(range(idx_min, idx_max), position=0, leave=True):
                detection = merged_detections_kept[idx]
                candidates = find_candidates(detections, detection, d_h, seen_dates=seen_dates, tolerance=TOLERANCE,
                                             min_p=MIN_P_MATES)
                # print(len(candidates))
                if len(candidates) < NB:
                    continue
                scores = {s: c for s, c in candidates.items()}
                best_m = best_matchup(scores, NB, d_h, TOLERANCE)
                if len(best_m) < NB:
                    continue

                matchup = [detection] + [candidates[s][j] for s, j in best_m]
                loc_worked, loc_res = locate(matchup, sound_model_h, 10)

                # if it didn't work because loc was not close enough, we try deleting a station
                if not loc_worked and type(loc_res) != list and len(matchup) > NB + 1:
                    to_del = np.argmax(loc_res.fun)  # index of maximum residual
                    matchup = matchup[:to_del] + matchup[to_del+1:]
                    loc_worked, loc_res = locate(matchup, sound_model_h, 10)


                # in case it worked, go further
                if loc_worked:
                    loc_worked, loc_res = locate(matchup, sound_model_g, 5, initial_pos=loc_res.x[1:])
                    if loc_worked:
                        det_times = [date] + [c[0] for c in matchup]
                        date_event = np.min(det_times) + datetime.timedelta(seconds=loc_res.x[0])

                        # add other stations
                        changed = False
                        loc_res_updated = copy.deepcopy(loc_res)
                        matchup_updated = copy.deepcopy(matchup)

                        seen_stations = set([c[-1] for c in matchup])
                        for s, c in candidates.items():
                            if s in seen_stations:
                                continue
                            try:
                                expected_time = date_event + datetime.timedelta(
                                    seconds=sound_model_h.get_sound_travel_time(s.get_pos(), loc_res_updated.x[1:]))
                            except:
                                continue
                            found = False
                            for chosen in c:
                                if np.abs(chosen[0] - expected_time).total_seconds() < 10:
                                    found = True
                                    break
                            if not found:
                                continue
                            matchup_updated = matchup_updated + [chosen]
                            loc_worked_new, loc_res_new = locate(matchup_updated, sound_model_h, 10)
                            if loc_worked_new:  # update location
                                changed = True
                                seen_stations.add(chosen[-1])
                                loc_res_updated = loc_res_new
                            else:  # rollback
                                matchup_updated = matchup_updated[:-1]

                        if changed:
                            loc_worked_new, loc_res_new = locate(matchup_updated, sound_model_g, 5, initial_pos=loc_res_updated.x[1:])
                            if loc_worked_new:
                                loc_res = loc_res_new
                                matchup = matchup_updated

                        d_from_cluster = np.sqrt(np.sum(np.square(np.array(loc_res.x[1:])-cluster_centroids[i])))
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

                        # note: we register one ISC event from the cluster
                        to_write = (f'{date_event.strftime("%Y%m%d_%H%M%S")},{loc_res.x[1]:.4f},{loc_res.x[2]:.4f},'
                                    f'{var[0]:.4f},{var[1]:.4f},{var[2]:.4f},{cluster[0]},{d_from_cluster:.4f}')
                        for d, _, _, s in matchup:
                            to_write += f',{s.name}-{s.date_start.year},{d.strftime("%Y%m%d_%H%M%S")}'
                        with open(RES_FILE, "a") as f:
                            f.write(to_write + "\n")