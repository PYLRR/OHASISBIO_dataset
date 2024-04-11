import datetime

import numpy as np
from scipy.signal import find_peaks

from GUI.widgets.mpl_canvas import MplCanvas
from GUI.widgets.spectral_view import SpectralView, MIN_SEGMENT_DURATION_S, QdatetimeToDatetime
from GUI.widgets.spectral_view_tissnet import SpectralViewTissnet
from utils.transformations.features_extractor import STFTFeaturesExtractor


class SpectralViewMatchupCheck(SpectralViewTissnet):
    def __init__(self, SpectralViewer, station, start_date=None, delta_view_s=MIN_SEGMENT_DURATION_S, *args, **kwargs):
        self.pointer_pos = None
        super().__init__(SpectralViewer, station, start_date, delta_view_s, *args, **kwargs)

    def _drawPointer(self):
        if self.pointer_pos is not None:
            self.mpl.axes.arrow(self.pointer_pos, self.manager.sampling_f/2-1, 0, -self.manager.sampling_f/20,
                                length_includes_head=True, head_width=10, head_length=10, color="red")

    def updatePlot(self):
        self._updateData()
        self._draw()
        self._drawPointer()

    def setPointerPos(self, value):
        self.pointer_pos = value
        self.updatePlot()

    def onclickGraph(self, click):
        if click.xdata is not None:
            delta = self.segment_length_doubleSpinBox.value()
            if self.pointer_pos and abs(click.xdata - self.pointer_pos) < delta/50:
                self.setPointerPos(None)
            else:
                self.setPointerPos(click.xdata)
            self.mpl.draw_idle()

    def onkeyGraph(self, key):
        if key.key == "enter":
            self.spectralViewer.validate()
        elif key.key == "backspace":
            self.spectralViewer.invalidate()
        else:
            super().onkeyGraph(key)