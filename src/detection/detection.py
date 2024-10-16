import math
from pathlib import Path

import glob2
import torch
from scipy import signal
import datetime
import os.path

import numpy as np
from scipy.signal import find_peaks, peak_widths
from torchvision.transforms import Resize
from tqdm import tqdm
import pickle
import tensorflow as tf
from line_profiler_pycharm import profile

from utils.data_reading.sound_data.station import StationsCatalog
from utils.training.TiSSNet import TiSSNet_torch as TiSSNet
from utils.training.embedder import Embedder, EmbedderSegmenter
from utils.transformations.features_extractor import STFTFeaturesExtractor


if __name__=="__main__":
    datasets_yaml = "/home/plerolland/Bureau/dataset.yaml"
    tissnet_checkpoint = "../../data/model_saves/TiSSNet/torch_save"
    embedder_checkpoint = "../../data/model_saves/embedder/torch_save_segmenter"

    # output files
    DELTA = datetime.timedelta(seconds=3600 / 0.98)  # /0.98 to get 1h segments
    TIME_RES = 500 / 936  # duration of each spectrogram pixel in seconds
    FREQ_RES = 240 / 256  # f of each spectrogram pixel in Hz
    TIME_RES_embedder = 0.5
    FREQ_RES_embedder = 0.5
    MAX_F = 120  # max tolerated f, which means we discard some pixel lines in case of IMS
    REQ_WIDTH = int(DELTA.total_seconds() / TIME_RES)
    REQ_WIDTH_embedder = int(DELTA.total_seconds() / TIME_RES_embedder)

    OVERLAP = 0.02  # overlap for models application (no link with STFT)
    STEP = (1 - OVERLAP) * DELTA

    TISSNET_PROMINENCE = 0.05
    ALLOWED_ERROR_S = 5
    MIN_HEIGHT = 0.05

    batch_size = 16

    stations_c = StationsCatalog(datasets_yaml).filter_out_undated().filter_out_unlocated()
    stft_computer = STFTFeaturesExtractor(None, vmin=-35, vmax=140)
    stft_computer_embedder = STFTFeaturesExtractor(None, vmin=60, vmax=140)

    # model = TiSSNet()
    # model.load_weights(tissnet_checkpoint)
    device = "cuda"
    model_det = torch.load(tissnet_checkpoint).to(device)
    model_embedder = torch.load(embedder_checkpoint).to(device)
    model_det.eval()
    model_embedder.eval()


    @profile
    def process_batch(batch):
        try:
            batch = np.array(batch)
        except:
            print("not rectangular array")
        batch = torch.from_numpy(batch).to(device)
        with torch.no_grad():
            res = model_det(batch).cpu().numpy()
        del batch
        torch.cuda.empty_cache()
        return res


    @profile
    def embed_batch(batch):
        try:
            batch = np.array(batch)
        except:
            print("not rectangular array")
        batch = torch.from_numpy(batch).to(device)
        with torch.no_grad():
            res = model_embedder(batch).cpu().numpy()
        del batch
        torch.cuda.empty_cache()
        return res

    @profile
    def main():
        for year in [2010]:
            Path(f"../../data/detections/{year}/").mkdir(parents=True, exist_ok=True)
            print(f"Starting detection on year {year}")
            stations = stations_c.ends_after(datetime.datetime(year, 1, 1) - datetime.timedelta(days=1))
            stations = stations.starts_before(datetime.datetime(year + 1, 1, 1) + datetime.timedelta(days=1))
            for station in stations:
                if "MAHY" in station.name:
                    continue
                embedding_path = f"../../data/detections/{year}/embedding_{year}_{station.name}-{station.date_start.year}"
                Path(embedding_path).mkdir(parents=True, exist_ok=True)
                results_file = f"../../data/detections/{year}/log_det_{year}_{station.name}-{station.date_start.year}.p"

                print(f"Processing station {station.name}-{station.date_start.year}")
                stft_computer.manager = station.get_manager()
                stft_computer.nperseg = round(stft_computer.manager.sampling_f / FREQ_RES)
                stft_computer.overlap = 1 - TIME_RES * stft_computer.manager.sampling_f / stft_computer.nperseg
                stft_computer_embedder.manager = station.get_manager()
                stft_computer_embedder.nperseg = int(stft_computer.manager.sampling_f / FREQ_RES_embedder)
                stft_computer_embedder.overlap = 1 - TIME_RES_embedder * stft_computer.manager.sampling_f / stft_computer.nperseg
                max_f_cut_line = round(MAX_F / FREQ_RES)
                max_f_cut_line_embedder = round(MAX_F / FREQ_RES_embedder)

                start = max(datetime.datetime(year, 1, 1), station.date_start + datetime.timedelta(days=1))
                end = min(datetime.datetime(year + 1, 1, 1), station.date_end - datetime.timedelta(days=1))
                steps = math.ceil((end - start) / STEP)
                start_idx = 0
                batch_dates, batch_process, batch_process_embedder = [], [], []

                existing_embedding_paths = glob2.glob(f"{embedding_path}/*.npy")
                if len(existing_embedding_paths) > 0:
                    last_date = np.max([datetime.datetime.strptime(p.split("/")[-1][:-4], "%Y%m%d_%H%M%S") for p in
                                        existing_embedding_paths])
                    del existing_embedding_paths
                    print(f'Station {station.name} already processed up to {last_date.strftime("%Y%m%d_%H%M%S")}')
                    start_idx = math.floor((last_date - start) / STEP)

                for i in tqdm(range(steps), smoothing=0.001):
                    if i < start_idx:
                        continue  # this is just to fill a part of tqdm

                    seg_start = start + i * STEP
                    seg_end = min(end, seg_start + DELTA)
                    if seg_start >= seg_end:
                        break

                    data_raw = station.get_manager().getSegment(seg_start, seg_end)
                    data = stft_computer._get_features(data_raw)[-1][:max_f_cut_line].astype(np.uint8)
                    data = (data[np.newaxis, :, :] / 255).astype(np.float32)
                    data_reduce = stft_computer_embedder._get_features(data_raw)[-1][:max_f_cut_line_embedder]
                    data_reduce = 2 * (data_reduce[np.newaxis, :, :] / 255 - 0.5).astype(np.float32)
                    del data_raw  # reclaim some RAM
                    if REQ_WIDTH * 1.02 > data.shape[
                        -1] > REQ_WIDTH * 0.98:  # we resize it to a standard number of time steps
                        data = Resize((128, REQ_WIDTH))(torch.from_numpy(data)).numpy()
                        data_reduce = Resize((64, REQ_WIDTH_embedder))(torch.from_numpy(data_reduce)).numpy()
                    batch_dates.append(seg_start)
                    batch_process.append(data)
                    batch_process_embedder.append(data_reduce)

                    if len(batch_process) == batch_size:
                        if batch_process[-1].shape != batch_process[0].shape or batch_process[-2].shape != batch_process[
                            -1].shape:
                            # last (and probably the one before because of overlaps) batch has a last element shorter than the others, we make three batches
                            rlastlast = process_batch(batch_process[-2])
                            rlast = process_batch(batch_process[-1])
                            rfirst = process_batch(batch_process[:-2])
                            res = list(rfirst) + [rlastlast] + [rlast]
                            del batch_process  # reclaim some RAM

                            rlastlast_embedder = embed_batch(batch_process_embedder[-2])
                            rlast_embedder = embed_batch(batch_process_embedder[-1])
                            rfirst_embedder = embed_batch(batch_process_embedder[:-2])
                            res_embedder = list(rfirst_embedder) + [rlastlast_embedder] + [rlast_embedder]
                            del batch_process_embedder  # reclaim some RAM
                        elif batch_process[0].shape != batch_process[1].shape or batch_process[1].shape != batch_process[
                            2].shape:
                            # first (and probably the one before because of overlaps) batch has a first element shorter than the others, we make three batches
                            rfirst = process_batch(batch_process[0])
                            rsecond = process_batch(batch_process[1])
                            rrest = process_batch(batch_process[2:])
                            res = [rfirst] + [rsecond] + list(rrest)
                            del batch_process  # reclaim some RAM

                            rfirst_embedder = embed_batch(batch_process_embedder[0])
                            rsecond_embedder = embed_batch(batch_process_embedder[1])
                            rrest_embedder = embed_batch(batch_process_embedder[2:])
                            res_embedder = [rfirst_embedder] + [rsecond_embedder] + list(rrest_embedder)
                            del batch_process_embedder  # reclaim some RAM
                        else:
                            res = process_batch(batch_process)
                            del batch_process  # reclaim some RAM
                            res_embedder = embed_batch(batch_process_embedder)
                            del batch_process_embedder  # reclaim some RAM

                        for i, (seg_start, r) in enumerate(zip(batch_dates, res)):
                            res_embedder_ = res_embedder[i].astype(np.float16)
                            res_embedder_ = res_embedder_[:, int(res_embedder_.shape[1] * OVERLAP / 2):-int(
                                res_embedder_.shape[1] * OVERLAP / 2)]
                            np.save(f'{embedding_path}/{(seg_start + DELTA * OVERLAP / 2).strftime("%Y%m%d_%H%M%S")}.npy',
                                    res_embedder_)
                            peaks = find_peaks(r, height=0, distance=ALLOWED_ERROR_S / TIME_RES,
                                               prominence=TISSNET_PROMINENCE)
                            time_s = peaks[0] * TIME_RES
                            peaks = [(seg_start + datetime.timedelta(seconds=time_s[j]), peaks[1]["peak_heights"][j]) for j
                                     in range(len(time_s)) if
                                     peaks[1]["peak_heights"][j] > MIN_HEIGHT and peaks[0][j] > REQ_WIDTH * OVERLAP / 2 and
                                     peaks[0][j] < REQ_WIDTH * (1 - OVERLAP / 2)]

                            with open(results_file, "ab") as f:
                                for i, (d, p) in enumerate(peaks):
                                    pickle.dump([d, p.astype(np.float16)], f)

                        batch_dates, batch_process, batch_process_embedder = [], [], []
    main()
