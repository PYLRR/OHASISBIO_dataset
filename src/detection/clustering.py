import datetime

import numpy as np

def get_valid(allowed_delta, delta, IDs):
    valid = {ID: {} for ID in IDs}
    for ID in IDs:
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
    for ID in IDs:
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

def get_delta(isc, max_allowed_delta):
    # compute distances of close in time events, expected run time ~2 min for 10,000 events with max_allowed_delta = 50 days
    idx_last_valid = 0
    delta_km = {}
    delta_d = {}
    delta = {}
    IDs = list(isc.items.keys())
    for i, (ID, isc_event) in enumerate(isc.items.items()):
        if i == 0:
            continue
        date, pos = isc_event.date, isc_event.get_pos()

        # peremption of clusters (by date)
        while idx_last_valid < i and isc[IDs[idx_last_valid]].date < date - datetime.timedelta(days=max_allowed_delta):
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

    return delta