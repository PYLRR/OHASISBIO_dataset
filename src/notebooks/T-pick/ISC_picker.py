from pathlib import Path

import pywt
from scipy import signal
from scipy.fft import fft
import datetime
import os.path

import numpy as np
from scipy.signal import find_peaks, peak_widths
from tqdm import tqdm
import pickle
import tensorflow as tf
from line_profiler_pycharm import profile

from utils.data_reading.catalogs.isc import ISC_file
from utils.data_reading.sound_data.station import StationsCatalog
from utils.physics.bathymetry_model import BathymetryModel
from utils.physics.sound_model import MonthlyGridSoundModel, HomogeneousSoundModel
from utils.training.TiSSNet import TiSSNet
from utils.transformations.features_extractor import STFTFeaturesExtractor

def apply_TiSSNet(batch, model):
    stft = tf.convert_to_tensor(batch, dtype=tf.uint8)
    return model.predict(stft, verbose=True, batch_size=32)

def apply_sta_lta(manager, time, delta, sta_delta, sf_to_mimic=240):
    sta_offset = int(sta_delta.total_seconds() * 2 * manager.sampling_f)
    pts_sta = manager.getSegment(time - delta - sta_delta, time + delta + sta_delta)
    pts_sta = np.square(pts_sta)
    lta = np.sqrt(np.mean(pts_sta))
    stas = np.sqrt(np.convolve(pts_sta, np.ones(sta_offset) / sta_offset, mode='valid')[::int(128 * manager.sampling_f/sf_to_mimic)][:-1])  # rescale the stas accordingly to the rescale of the spectrograms, given their standard sampling frequency is 240 Hz
    return stas / lta


@profile
def compute_peaks(time_series, station, time, global_welch, min_height, distance, height_ratio_for_width, time_res, delta, prominence=None):
    global_energy = np.sum(global_welch[1][1:])

    peaks = find_peaks(time_series, height=min_height, distance=distance, prominence=prominence)
    width = peak_widths(time_series, peaks[0], height_ratio_for_width)[0]
    time_s = peaks[0] * time_res

    date_bounds = [(peaks[0][i] - width[i] / 2, peaks[0][i] + width[i] / 2) for i in range(len(width))]
    date_bounds = [[time + datetime.timedelta(seconds=d * time_res) - delta for d in d_b] for d_b in date_bounds]
    data = [station.manager.getSegment(d[0], d[1]) for d in date_bounds]
    max_energy_time_s = np.array([time_s[i] - width[i] / 2 + np.argmax(np.abs(data[i]))/station.manager.sampling_f for i in range(len(data))])
    welch = [signal.welch(d, station.manager.sampling_f, 'flattop', 64, scaling='spectrum') for d in data]
    SNR = np.array([np.sqrt(np.sum(w[1][1:]) / global_energy) for w in welch])

    peaks = {"time_s": time_s, "max_energy_time_s": max_energy_time_s,
             "width_s": width * time_res, "height": peaks[1]["peak_heights"], "welch": welch, "SNR": SNR}
    return peaks

if __name__=="__main__":
    @profile
    def main():
        for year in [2018]:
            print(f"STARTING {year} at {datetime.datetime.now()}")
            # input files
            datasets_yaml = "/home/plerolland/Bureau/dataset.yaml"
            isc_file = f"/home/plerolland/Bureau/catalogs/ISC/eqk_isc_{year}.txt"
            velocities_file = "../../../data/geo/velocities_grid.pkl"
            bathy_file = "../../../data/geo/GEBCO_2023_sub_ice_topo.nc"
            tissnet_checkpoint = "../../../data/model_saves/TiSSNet/all/cp-0022.ckpt"

            # output files
            to_process_file = f"../../../data/T-pick/{year}/to_process_det"
            results_file = f"../../../data/T-pick/{year}/log_det"

            DELTA = datetime.timedelta(seconds=100)
            TIME_RES = 128 / 240  # duration of each spectrogram pixel in seconds
            WIDTH = int(2 * DELTA.total_seconds() / TIME_RES)  # width of the spectrograms in seconds
            TISSNET_PROMINENCE = 0.05
            ALLOWED_ERROR_S = 5  # time distance allowed between two peaks in the probabilities distributions
            STA_DELTA = datetime.timedelta(seconds=5)
            STALTA_PROMINENCE = 0.05
            PEAKS_REL_HEIGHT = 0.75

            stations = StationsCatalog(datasets_yaml).filter_out_undated().filter_out_unlocated()
            stations = stations.ends_after(datetime.datetime(year,1,1) - datetime.timedelta(days=1))
            stations = stations.starts_before(datetime.datetime(year+1,1,1) + datetime.timedelta(days=1))
            isc = ISC_file(isc_file)
            IDs = list(isc.items.keys())
            sound_model = HomogeneousSoundModel()
            bathy_model = BathymetryModel(bathy_file, lat_bounds=[-75, 35], lon_bounds=[-20, 180])
            stft_computer = STFTFeaturesExtractor(None, vmin=-35, vmax=140)
            model = TiSSNet()
            model.load_weights(tissnet_checkpoint)


            to_process = []
            # make the parent directory
            Path('/'.join(to_process_file.split("/")[:-1])).mkdir(parents=True, exist_ok=True)
            if os.path.isfile(to_process_file) and os.path.getsize(to_process_file) > 0:
                with open(to_process_file, 'rb') as f:
                    to_process = pickle.load(f)
            else:
                for ID in tqdm(IDs):
                    event = isc[ID]
                    candidates = stations.by_date_propagation(event, sound_model, delta=DELTA)
                    if len(candidates) == 0:
                        continue
                    for station, time_of_arrival in candidates:
                        to_process.append((ID, station, time_of_arrival))
                with open(to_process_file, 'wb') as f:
                    pickle.dump(to_process, f)


            batch_size = 1024

            idx = 0
            detections = {}
            batch_n = 0
            if os.path.isfile(results_file) and os.path.getsize(results_file) > 0:
                with open(results_file, 'rb') as f:
                    idx, detections = pickle.load(f)

            print(f"Batches left to process : {int(np.ceil((len(to_process)-idx) / batch_size))} "
                  f"(total samples done {idx} / {len(to_process)})")
            while idx < len(to_process):
                batch = []
                batch_PSDs = []
                batch_meta = []  # contains lists of (ID, station, time_of_arrival)
                batch_n += 1
                for _ in tqdm(range(min(len(to_process) - idx, batch_size)), desc="loading data"):
                    _, station, time = to_process[idx]
                    batch_meta.append(to_process[idx])
                    idx += 1
                    manager = station.get_manager()

                    stft_computer.manager = manager
                    data = manager.getSegment(time - DELTA, time + DELTA)
                    batch.append(stft_computer._get_features(data)[-1].astype(np.uint8))
                    batch_PSDs.append(signal.welch(data, manager.sampling_f, 'flattop', 64, scaling='spectrum'))
                    batch[-1] = np.expand_dims(batch[-1], axis=-1)
                    batch[-1] = tf.image.resize(batch[-1], size=(128, WIDTH))
                res = apply_TiSSNet(batch, model)

                for i, r in tqdm(enumerate(res), desc="processing model output"):
                    ID, station, time = batch_meta[i]
                    if isc[ID].get_pos()[1] < -50:
                        # some isc events are not well sorted and we may find absurd entries
                        continue

                    # TiSSNet
                    tissnet_peaks = compute_peaks(r, station, time, batch_PSDs[i], 0, ALLOWED_ERROR_S / TIME_RES,
                                                  PEAKS_REL_HEIGHT, time_res=TIME_RES, delta=DELTA,
                                                  prominence=TISSNET_PROMINENCE)

                    # STA / LTA
                    sta_lta = apply_sta_lta(station.manager, time, DELTA, STA_DELTA)
                    sta_lta_peaks = compute_peaks(sta_lta, station, time, batch_PSDs[i], 0,
                                                  2 * ALLOWED_ERROR_S / TIME_RES, PEAKS_REL_HEIGHT, time_res=TIME_RES,
                                  delta=DELTA, prominence=STALTA_PROMINENCE)

                    if ID not in detections:
                        detections[ID] = []

                    bathy_profile = bathy_model.get_bathymetry_along_path(isc[ID].get_pos(), station.get_pos())

                    detections[ID].append((station.light_copy(), time, batch_PSDs[i], tissnet_peaks, sta_lta_peaks, np.max(bathy_profile)))

                with open(results_file, 'wb') as f:
                    pickle.dump((idx, detections), f)

        print(f"ENDING at {datetime.datetime.now()}")
    main()
