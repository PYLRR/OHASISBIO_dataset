import datetime
import pickle
from pathlib import Path

from tqdm import tqdm

from utils.data_reading.catalogs.ISC import ISC_file
from utils.data_reading.sound_data.station import StationsCatalog
from utils.transformations.features_extractor import STFTFeaturesExtractor


if __name__=="__main__":
    year = 2019

    # input files
    isc_file = f"/home/plerolland/Bureau/catalogs/ISC/eqk_isc_{year}.txt"
    velocities_file = "../../../data/geo/velocities_grid.pkl"
    datasets_yaml = "/home/plerolland/Bureau/dataset.yaml"
    dets_file = f"../../../data/detections/log_det_{year}"
    matchups_file = f"../../../data/detections/matchups_isc_{year}"

    # output directory
    dest = f"/home/plerolland/Bureau/ISC_matchup_check/{year}"


    MIN_PROBA = 0.3
    MIN_PROBA_PER_MATCHUP = 0.6
    MIN_PROBA_NOISY = 0.6
    MAX_DETECTIONS_PER_SEG = 5
    TIME_RES = 100 / 186
    DELTA = datetime.timedelta(seconds=100)
    ALLOWED_DEVIATION = datetime.timedelta(seconds=75)

    isc = ISC_file(isc_file)
    with open(dets_file, 'rb') as f:
        detections = pickle.load(f)[1]



    stations = StationsCatalog(datasets_yaml).filter_out_undated().filter_out_unlocated()
    stations = stations.ends_after(datetime.datetime(year, 1, 1) - datetime.timedelta(days=1))
    stations = stations.starts_before(datetime.datetime(year + 1, 1, 1) + datetime.timedelta(days=1))

    with open(matchups_file, "rb") as f:
        selected_matchups = pickle.load(f)



    time_res = 0.5
    freq_res = 120/128
    STFT_computer = STFTFeaturesExtractor(None, f_min=0, f_max=120, vmin=70, vmax=100)


    seen_stations = []

    for ID, matchup in tqdm(selected_matchups.items()):
        available_stations = [d[0] for d in detections[ID]]
        for i, station in enumerate(available_stations):
            # try to get station from history if it can be found there
            if station in seen_stations:
                station = seen_stations[seen_stations.index(station)]
            else:
                seen_stations.append(station)

            STFT_computer.manager = station.get_manager()
            STFT_computer.nperseg = round(STFT_computer.manager.sampling_f / freq_res)
            STFT_computer.overlap = 1 - time_res * STFT_computer.manager.sampling_f / STFT_computer.nperseg
            folder = f"{dest}/{station.name}"
            Path(folder).mkdir(parents=True, exist_ok=True)

            det = matchup[station]["time_s"] if station in matchup else -1

            center = detections[ID][i][1]
            start, end = center - DELTA, center + DELTA
            if not Path(f"{folder}/{ID}_{det}.png").exists():
                STFT_computer.save_features(start, end, f"{folder}/{ID}_{det:.2f}.png")