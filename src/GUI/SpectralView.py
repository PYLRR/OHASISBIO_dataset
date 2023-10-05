import datetime

import numpy as np
import scipy
from PySide6 import QtWidgets, QtGui
from PySide6.QtWidgets import QLabel, QSlider, \
    QDateTimeEdit, QDoubleSpinBox
from PySide6.QtCore import QTimer
from PySide6.QtCore import (QDate, QDateTime, QTime, Qt)
from playsound import playsound

from GUI.Utilities.MplCanvas import MplCanvas
from utils.FeaturesExtractor import STFTFeaturesExtractor


# Window showing the spectrogram of a .wav folder
class SpectralView(QtWidgets.QWidget):
    MIN_SPECTRO_DURATION = datetime.timedelta(seconds=10)

    """
    Custom Qt Widget to explore a .wav folder by viewing spectrograms
    """
    def __init__(self, manager, SpectralViews=None, event=None, delta_view_s=30, *args, **kwargs):
        super(SpectralView, self).__init__(*args, **kwargs)

        # divisor of the current max frequency shown on the spectrograms
        self.freq_range = [0, int(manager.sampling_f / 2)]

        self.manager = manager
        self.STFT_computer = STFTFeaturesExtractor(self.manager, f_min=self.freq_range[0], f_max=self.freq_range[1])

        # timer used to prevent manual resizing of the window to cause lagging by drawing spectro too often
        self.resize_timer = QTimer()

        self.layout = QtWidgets.QHBoxLayout(self)
        self.setLayout(self.layout)

        self.control_layout = QtWidgets.QVBoxLayout()
        self.layout.addLayout(self.control_layout)
        self.label = QLabel(self)
        self.label.setText(manager.path.split("/")[-1])
        if len(manager.path.split("/")[-1]) == 0:
            self.label.setText(manager.path.split("/")[-2])
        font = QtGui.QFont()
        font.setBold(True)
        self.label.setFont(font)
        self.control_layout.addWidget(self.label)

        self.segment_length_layout = QtWidgets.QGridLayout()
        self.segment_length_layout.setAlignment(Qt.AlignmentFlag.AlignBottom)
        self.control_layout.addLayout(self.segment_length_layout)
        self.segment_length_label = QLabel(self)
        self.segment_length_label.setText("Segment duration (s)")
        self.segment_length_doubleSpinBox = QDoubleSpinBox(self)
        self.segment_length_doubleSpinBox.setMinimum(60)
        self.segment_length_doubleSpinBox.setMaximum(2**20)
        self.segment_length_doubleSpinBox.setValue(2*delta_view_s)
        self.segment_length_slider = QSlider(self)
        self.segment_length_slider.setMinimum(60)
        self.segment_length_slider.setMaximum(36000)
        self.segment_length_slider.setOrientation(Qt.Horizontal)
        self.segment_length_layout.addWidget(self.segment_length_label, 0, 0, 1, 1)
        self.segment_length_layout.addWidget(self.segment_length_doubleSpinBox, 1, 0, 1, 1)
        self.segment_length_layout.addWidget(self.segment_length_slider, 2, 0, 1, 3)

        self.segment_date_layout = QtWidgets.QGridLayout()
        self.segment_date_layout.setAlignment(Qt.AlignmentFlag.AlignBottom)
        self.control_layout.addLayout(self.segment_date_layout)
        self.segment_date_label = QLabel(self)
        self.segment_date_label.setText("Segment date")
        self.segment_date_dateTimeEdit = QDateTimeEdit(self)
        self.segment_date_dateTimeEdit.setInputMethodHints(Qt.ImhPreferNumbers)
        self.segment_date_dateTimeEdit.setDisplayFormat("yyyy/MM/dd hh:mm:ss")
        self.segment_date_dateTimeEdit.setDateTime(DatetimeToQdatetime(manager.dataset_start, day_offset=1))
        if event is not None:
            self.segment_date_dateTimeEdit.setDateTime(DatetimeToQdatetime(event))
        self.segment_date_slider = QSlider(self)
        self.segment_date_slider.setOrientation(Qt.Horizontal)
        self.segment_date_layout.addWidget(self.segment_date_label, 0, 0, 1, 1)
        self.segment_date_layout.addWidget(self.segment_date_dateTimeEdit, 1, 0, 1, 1)
        self.segment_date_layout.addWidget(self.segment_date_slider, 2, 0, 1, 3)

        self.mpl = MplCanvas(self, width=12, height=6, dpi=120)
        self.mpl.axes.figure.subplots_adjust(left=0.05, right=0.95, bottom=0.2, top=0.9)
        self.layout.addWidget(self.mpl)

        # this connection is used to prevent the user from asking for out of bound data
        self.segment_length_doubleSpinBox.valueChanged.connect(self.updateDatesBounds)
        self.segment_length_doubleSpinBox.valueChanged.connect(self.updateSegmentLengthSlider)
        self.segment_length_slider.sliderReleased.connect(self.updateSegmentLengthEditor)
        self.segment_length_doubleSpinBox.valueChanged.connect(self.updatePlot)
        self.segment_date_dateTimeEdit.dateTimeChanged.connect(self.updatePlot)

        self.segment_date_dateTimeEdit.dateTimeChanged.connect(self.updateSlider)
        self.segment_date_slider.sliderReleased.connect(self.updateDateTimeEdit)
        self.updateDatesBounds()
        self.updateSlider()
        self.updatePlot()
        self.layout.setStretch(1, 5)

    def updateSegmentLengthEditor(self):
        self.segment_length_doubleSpinBox.setValue(self.segment_length_slider.value())
    def updateSegmentLengthSlider(self):
        self.segment_length_slider.setValue(self.segment_length_doubleSpinBox.value())

    def resizeEvent(self, event):
        self.resize_timer.stop()
        del self.resize_timer
        self.resize_timer = QTimer()
        self.resize_timer.timeout.connect(self.updatePlot)
        # 100 ms after window resize, we redraw the plot, if no other resize happens
        self.resize_timer.start(100)

    def updateSlider(self):
        self.segment_date_slider.setValue((QdatetimeToDatetime(self.segment_date_dateTimeEdit.date(),
            self.segment_date_dateTimeEdit.time()) - self.manager.dataset_start).total_seconds())

    def updateDateTimeEdit(self):
        delta = self.segment_length_doubleSpinBox.value() / 2 + 1  # 1 s of margin
        delta = datetime.timedelta(seconds=max(delta, self.MIN_SPECTRO_DURATION.total_seconds()))
        self.segment_date_dateTimeEdit.setDateTime(DatetimeToQdatetime(
            self.manager.dataset_start + delta + datetime.timedelta(seconds=self.segment_date_slider.value())))

    def updateDatesBounds(self):
        delta = self.segment_length_doubleSpinBox.value() / 2 + 1  # 1 s of margin
        delta = datetime.timedelta(seconds=max(delta, self.MIN_SPECTRO_DURATION.total_seconds()))
        start = self.manager.dataset_start + delta
        end = self.manager.dataset_end - delta
        self.segment_date_dateTimeEdit.setMinimumDateTime(DatetimeToQdatetime(start))
        self.segment_date_dateTimeEdit.setMaximumDateTime(DatetimeToQdatetime(end))
        self.segment_date_slider.setMinimum(0)
        self.segment_date_slider.setMaximum((end-start).total_seconds())

    def updatePlot(self):
        self._updateData()
        self._draw()

    def _updateData(self):
        # in case we got here because the window has been resized, stop the timer to avoid triggering it again
        self.resize_timer.stop()

        self.STFT_computer.f_min, self.STFT_computer.f_max = self.freq_range[0], self.freq_range[1]

        segment_center = QdatetimeToDatetime(self.segment_date_dateTimeEdit.date(),
                                             self.segment_date_dateTimeEdit.time())

        delta = datetime.timedelta(seconds=self.segment_length_doubleSpinBox.value() / 2)


        (f, t, spectro) = self.STFT_computer.get_features(segment_center - delta, segment_center + delta)
        extent, aspect = self.STFT_computer._get_extent_and_aspect((f, t, spectro))
        # put central time at 0
        extent = list(extent)
        extent[0], extent[1] = extent[0] - delta.total_seconds(), extent[1] - delta.total_seconds()

        w = self.mpl.width()
        h = self.mpl.height()
        self.mpl.axes.cla()
        self.mpl.axes.imshow(spectro, aspect=aspect * ((h+10) / w), extent=extent)
        self.mpl.axes.set_xlabel('t (s)')
        self.mpl.axes.set_ylabel('f (Hz)')

    def _draw(self):
        self.mpl.draw()

    def onclickGraph(self, click):
        if click.xdata is not None:
            segment_center = QdatetimeToDatetime(self.segment_date_dateTimeEdit.date(),
                                                 self.segment_date_dateTimeEdit.time())
            segment_center += datetime.timedelta(seconds=click.xdata)
            self.segment_date_dateTimeEdit.setDateTime(DatetimeToQdatetime(segment_center))
            self.updatePlot()

    def onkeyGraph(self, key):
        segment_center = QdatetimeToDatetime(self.segment_date_dateTimeEdit.date(),
                                             self.segment_date_dateTimeEdit.time())
        delta = datetime.timedelta(seconds=self.segment_length_doubleSpinBox.value() / 2)
        if key.key == 'right':
            # go to right
            segment_center += delta
        if key.key == 'left':
            # go to left
            segment_center -= delta
        if key.key == '+':
            # zoom
            delta /= 2
        if key.key == '-':
            # dezoom
            delta *= 2
        if key.key == 'enter':
            # play the sound with increased frequency
            data = self.manager.getSegment(segment_center - delta, segment_center + delta)
            data = data / np.max(np.abs(data))

            to_write = np.int16(32767 * data / np.max(np.abs(data)))
            scipy.io.wavfile.write("/tmp/out.wav", int(self.manager.sampling_f*20), to_write)  # increased f to render better
            playsound("/tmp/out.wav")
            #os.remove('/tmp/out.wav')

        if key.key == "up":
            d = min(self.manager.sampling_f/2 - self.freq_range[1],
                        0.1 * self.manager.sampling_f / 2)
            self.freq_range[0] += d
            self.freq_range[1] += d
            self.updatePlot()

        if key.key == "down":
            d = min(self.freq_range[0],
                        0.1 * self.manager.sampling_f / 2)
            self.freq_range[0] -= d
            self.freq_range[1] -= d
            self.updatePlot()

        if key.key == "*":
            d = 0.05 * (self.freq_range[1]-self.freq_range[0])
            self.freq_range[1] -= d
            self.updatePlot()

        if key.key == "/":
            delta_top = min(0.05 * (self.freq_range[1] - self.freq_range[0]),
                            self.manager.sampling_f / 2 - self.freq_range[1])
            delta_bot = min(0.05 * (self.freq_range[1] - self.freq_range[0]),
                            self.freq_range[0])
            self.freq_range[0] -= delta_bot
            self.freq_range[1] += delta_top
            self.updatePlot()

            # shift + enter = set all SpectralView windows to the current date of this SpectralView
        if key.key == "shift+enter":
            segment_center = QdatetimeToDatetime(self.segment_date_dateTimeEdit.date(),
                                                 self.segment_date_dateTimeEdit.time())
            for spectralview in self.spectralViews.SpectralViews:
                spectralview.changeDate(segment_center)
                spectralview.changeSegmentLength(self.segment_length_doubleSpinBox.value())

        self.segment_date_dateTimeEdit.setDateTime(DatetimeToQdatetime(segment_center))
        self.segment_length_doubleSpinBox.setValue(delta.total_seconds()*2)
        self.updatePlot()

    def onscroll(self, scroll):
        pass

    def changeDate(self, new_date):
        new_date = new_date + datetime.timedelta(seconds=round(new_date.microsecond/10**6))
        new_date = new_date.replace(microsecond=0)
        self.segment_date_dateTimeEdit.setDateTime(DatetimeToQdatetime(new_date))

    def changeSegmentLength(self, new_length):
        self.segment_length_doubleSpinBox.setValue(new_length)


def QdatetimeToDatetime(qdate, qtime):
    return datetime.datetime(qdate.year(), qdate.month(), qdate.day(),
                             qtime.hour(), qtime.minute(), qtime.second())


def DatetimeToQdatetime(datetime, day_offset=0):
    return QDateTime(
        QDate(datetime.year, datetime.month, datetime.day + day_offset),
        QTime(datetime.hour, datetime.minute, datetime.second))