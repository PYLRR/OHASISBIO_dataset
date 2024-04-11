import pywt
from scipy import signal
from scipy.fft import fft

if __name__=="__main__":
    import datetime
    import os.path

    import numpy as np
    from scipy.signal import find_peaks, peak_widths
    from tqdm import tqdm
    import pickle
    import tensorflow as tf

    from utils.data_reading.catalogs.isc import ISC_file
    from utils.data_reading.sound_data.station import StationsCatalog
    from utils.physics.sound_model import MonthlyGridSoundModel, HomogeneousSoundModel
    from utils.training.keras_models import TiSSNet
    from utils.transformations.features_extractor import STFTFeaturesExtractor

    year = 2018
    datasets_yaml = "/home/plerolland/Bureau/datasets.yaml"
    isc_file = f"/home/plerolland/Bureau/catalogs/ISC/eqk_isc_{year}.txt"
    velocities_file = "../../../data/geo/velocities_grid.pkl"
    tissnet_checkpoint = "../../../data/model_saves/TiSSNet/all/cp-0022.ckpt"
    to_process_file = f"../../../data/detections/to_process_det_{year}"
    results_file = f"../../../data/detections/log_det_0211_{year}"
    DELTA = datetime.timedelta(seconds=200)
    MIN_PROBA = 0.05  # minimum value of the output of the segmenter model to record it
    TIME_RES = 100 / 186  # duration of each spectrogram pixel in seconds
    SNR_ST_DELTA = int(10 / TIME_RES)
    ALLOWED_ERROR_S = 20  # time distance allowed between two peaks in the probabilities distributions

    stations = StationsCatalog(datasets_yaml).filter_out_undated().filter_out_unlocated()
    stations = stations.ends_after(datetime.datetime(year,1,1) - datetime.timedelta(days=1))
    stations = stations.starts_before(datetime.datetime(year+1,1,1) + datetime.timedelta(days=1))
    isc = ISC_file(isc_file)
    IDs = list(isc.items.keys())
    sound_model = HomogeneousSoundModel()
    stft_computer = STFTFeaturesExtractor(None, vmin=-35, vmax=140)
    model = TiSSNet()
    model.load_weights(tissnet_checkpoint)


    to_process = []
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


    RAM_per_spectro = 128 * DELTA.total_seconds() * 2 / TIME_RES
    AVAILABLE_RAM = 100_000_000
    batch_size = int(AVAILABLE_RAM / RAM_per_spectro)

    idx = 0
    detections = {}
    width = int(2 * DELTA.total_seconds() / TIME_RES)
    batch_n = 0
    if os.path.isfile(results_file) and os.path.getsize(results_file) > 0:
        with open(results_file, 'rb') as f:
            idx, detections = pickle.load(f)

    print(f"Batches left to process : {int(np.ceil((len(to_process)-idx) / batch_size))} "
          f"(total samples done {idx} / {len(to_process)})")
    to_save_ID = set()
    while idx < len(to_process):
        batch = []
        batch_PSDs = []
        batch_meta = []
        batch_n += 1
        for _ in tqdm(range(min(len(to_process) - idx, batch_size)), desc="loading data"):
            _, station, time = to_process[idx]
            batch_meta.append(to_process[idx])
            idx += 1
            manager = station.get_manager()
            stft_computer.manager = manager
            data = manager.getSegment(time - DELTA, time + DELTA)
            batch.append(stft_computer._get_features(data)[-1].astype(np.uint8))
            batch_PSDs.append(signal.welch(data, 240, 'flattop', 64, scaling='spectrum'))
            batch[-1] = np.expand_dims(batch[-1], axis=-1)
            batch[-1] = tf.image.resize(batch[-1], size=(128, width))
        batch = tf.convert_to_tensor(batch, dtype=tf.uint8)
        res = model.predict(batch, verbose=True, batch_size=8)
        for i, r in tqdm(enumerate(res), desc="processing model output"):
            peaks = find_peaks(r, height=MIN_PROBA, distance=ALLOWED_ERROR_S / TIME_RES)
            widths = peak_widths(r, peaks[0], 0.5)
            peaks[1]['peak_widths'] = widths
            date_centers = [batch_meta[i][2] - DELTA + datetime.timedelta(seconds=p*TIME_RES) for p in peaks[0]]
            widths = [datetime.timedelta(seconds=w*TIME_RES) for w in widths[0]]
            date_bounds = [(date_centers[i] - widths[i]/2, date_centers[i] + widths[i]/2) for i in range(len(widths))]
            data = [batch_meta[i][1].manager.getSegment(d[0], d[1]) for d in date_bounds]
            welch = [signal.welch(d, 240, 'flattop', 64, scaling='spectrum') for d in data]
            peaks[1]['welch'] = welch
            ID = batch_meta[i][0]
            if ID not in detections:
                detections[ID] = []
            detections[ID].append((batch_meta[i][1].light_copy(), batch_meta[i][2], batch_PSDs[i], peaks))
        with open(results_file, 'wb') as f:
            pickle.dump((idx, detections), f)
