import numpy as np
import torch
from scipy.signal import find_peaks
from skimage.transform import resize

from GUI.widgets.mpl_canvas import MplCanvas
from GUI.widgets.spectral_view import SpectralView, MIN_SEGMENT_DURATION_S
from utils.transformations.features_extractor import STFTFeaturesExtractor


class SpectralViewTissnet(SpectralView):
    """ Spectral view enabling, with shift+enter, to apply TiSSNet on the current spectrogram.
    """
    def __init__(self, SpectralViewer, station, date=None, delta_view_s=MIN_SEGMENT_DURATION_S, *args, **kwargs):
        """ Initialize the SpectralView.
        :param SpectralViewer: The parent SpectralViews window.
        :param station: A Station instance.
        :param date: Initial datetime on which to focus the widget. If None, the start of the data will be chosen.
        :param delta_view_s: Initial half duration of the shown spectrogram, in s.
        :param args: Supplementary arguments for the widget as a PySide widget.
        :param kwargs: Supplementary key arguments for the widget as a PySide widget.
        """
        super().__init__(SpectralViewer, station, date, delta_view_s, *args, **kwargs)
        self.initMpl()
        self.mpl_layout.addWidget(self.mpl_models)
        self.STFT_computer = STFTFeaturesExtractor(self.manager, f_min=self.freq_range[0], f_max=self.freq_range[1],
                                                   cmap="jet", vmin=70, vmax=100, nperseg=1024, overlap=0.9)
        # we need a STFT computer using the parameters with which models have been trained
        self.STFT_computer_for_TiSSNet = STFTFeaturesExtractor(self.manager, f_min=0, f_max=240,
                                                               vmin=-35, vmax=140, nperseg=256, overlap=0.5)
        self.STFT_computer_for_embedder = STFTFeaturesExtractor(self.manager, f_min=0, f_max=240,
                                                               vmin=60, vmax=140, nperseg=480, overlap=0.75)

    def onkeyGraph(self, key):
        """ Checks if the TiSSNet shortcut has been pressed.
        :param key: The key pressed.
        :return: None.
        """
        if key.key == "shift+enter":
            self.processTissnet()
        elif key.key == "alt+enter":
            self.spectralViewer.associate(self)
        else:
            super().onkeyGraph(key)

    def initMpl(self):
        """ Initialize the Matplotlib widget.
        :return: None
        """
        self.mpl_models = MplCanvas(self)
        self.mpl_models.axes.figure.subplots_adjust(left=0.05, right=0.97, bottom=0, top=0.94)
        self.mpl_models.axes.axis('off')
        self.mpl_models.setFixedHeight(40)
        self.mpl_models.setVisible(False)

    def processTissnet(self):
        """ Apply TiSSNet on the current spectrogram and adds the result below it with a colormap.
        :return: None.
        """
        if self.spectralViewer.detection_model is None:
            print("Trying to use TiSSNet but it has not been loaded")
            return

        start, end = self.getTimeBounds()
        (f, t, spectro) = self.STFT_computer_for_TiSSNet.get_features(start, end)
        spectro = (spectro[np.newaxis, :, :] / 255).astype(np.float32)
        spectro = resize(spectro, (1, 128, spectro.shape[2]))
        spectro = torch.from_numpy(spectro)
        with torch.no_grad():
            res = self.spectralViewer.detection_model(spectro).numpy()

        peaks = find_peaks(res, height=0.1, distance=10)
        print(peaks)

        self.showModelResult(res)

    def processEmbedder(self, margin, keep_margin=False):
        """ Apply the embedder.
        :param margin: margin to take to avoid border effects.
        :param keep_margin: if True, do not remove the margins from output.
        :return: Embedding of the data.
        """
        start, end = self.getTimeBounds()
        (f, t, spectro) = self.STFT_computer_for_embedder.get_features(start-margin, end+margin)
        spectro = 2 * (spectro[np.newaxis, :, :] / 255 - 0.5).astype(np.float32)
        spectro = resize(spectro, (1, 64, spectro.shape[2]))
        spectro = torch.from_numpy(spectro)
        with torch.no_grad():
            res = self.spectralViewer.association_model(spectro).numpy()
        res = res if keep_margin else res[:,int(margin.total_seconds()*2):-int(margin.total_seconds()*2)]
        return res

    def showModelResult(self, to_show):
        """ Shows a 1D time series as a small heat map.
        :param to_show: The 1D data to show.
        :return: None.
        """
        self.mpl_layout.removeWidget(self.mpl_models)
        self.initMpl()
        self.mpl_layout.addWidget(self.mpl_models)

        self.mpl_models.axes.imshow(to_show.reshape((1, -1)), aspect="auto", vmin=0, vmax=1)
        self.mpl_models.setVisible(True)