import numpy as np
from scipy.signal import find_peaks

from GUI.widgets.mpl_canvas import MplCanvas
from GUI.widgets.spectral_view import SpectralView, MIN_SEGMENT_DURATION_S
from utils.transformations.features_extractor import STFTFeaturesExtractor


class SpectralViewTissnet(SpectralView):
    def __init__(self, SpectralViewer, station, start_date=None, delta_view_s=MIN_SEGMENT_DURATION_S, *args, **kwargs):
        super().__init__(SpectralViewer, station, start_date, delta_view_s, *args, **kwargs)
        self.initMpl()
        self.mpl_layout.addWidget(self.mpl_tissnet)
        self.STFT_computer = STFTFeaturesExtractor(self.manager, f_min=self.freq_range[0], f_max=self.freq_range[1],
                                                   cmap="jet", vmin=70, vmax=100, nperseg=1024, overlap=0.9)

    def onkeyGraph(self, key):
        if key.key == "shift+enter":
            self.processTissnet()
        else:
            super().onkeyGraph(key)

    def initMpl(self):
        self.mpl_tissnet = MplCanvas(self)
        self.mpl_tissnet.axes.figure.subplots_adjust(left=0.05, right=0.97, bottom=0, top=0.94)
        self.mpl_tissnet.axes.axis('off')
        self.mpl_tissnet.setFixedHeight(40)
        self.mpl_tissnet.setVisible(False)
    def processTissnet(self):
        if self.spectralViewer.model is None:
            print("Trying to use TiSSNet but it does not have been loaded")
            return
        self.mpl_layout.removeWidget(self.mpl_tissnet)
        self.initMpl()
        self.mpl_layout.addWidget(self.mpl_tissnet)

        start, end = self.getTimeBounds()
        (f, t, spectro) = self.STFT_computer.get_features(start, end)
        spectro = spectro.astype(np.uint8)
        spectro = spectro[:128]
        res = self.spectralViewer.model.predict(spectro.reshape((1, 128, -1)), verbose=False)[0]

        self.mpl_tissnet.axes.imshow(res.reshape((1, -1)), aspect="auto", vmin=0, vmax=1)
        self.mpl_tissnet.setVisible(True)

        peaks = find_peaks(res, height=0.1, distance=10)
        print(peaks)
