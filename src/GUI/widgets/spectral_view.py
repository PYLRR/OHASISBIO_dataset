import datetime
import os

import numpy as np
import scipy
from PySide6 import QtWidgets, QtGui
from PySide6.QtWidgets import QLabel, QSlider, \
    QDateTimeEdit, QDoubleSpinBox
from PySide6.QtCore import (QDate, QDateTime, QTime, Qt, QTimer)
from playsound import playsound

from GUI.widgets.mpl_canvas import MplCanvas
from utils.data_reading.features_extractor import STFTFeaturesExtractor


class SpectralView(QtWidgets.QWidget):
    """ Widget to explore a .wav folder by viewing spectrograms
    """
    # minimal duration of the spectrograms shown, in s
    MIN_SEGMENT_DURATION_S = 30
    # maximal duration of the spectrograms shown, in s
    MAX_SEGMENT_DURATION_S = 36_000

    def __init__(self, SpectralViews, manager, start_date=None, delta_view_s=MIN_SEGMENT_DURATION_S, *args, **kwargs):
        """ Constructor initializing various parameters and setting up the interface.
        :param SpectralViews: The parent SpectralViews window.
        :param manager: A SoundFilesManager instance.
        :param start_date: Initial datetime on which to focus the widget. If None, the start will be chosen.
        :param delta_view_s: Initial half duration of the shown spectrogram, in s.
        :param args: Supplementary arguments for the widget as a PySide widget.
        :param kwargs: Supplementary key arguments for the widget as a PySide widget.
        """
        super(SpectralView, self).__init__(*args, **kwargs)
        self.spectralViews = SpectralViews
        self.manager = manager

        # min and max shown frequencies
        self.freq_range = [0, int(manager.sampling_f / 2)]

        # features extractor used to compute the spectrograms
        self.STFT_computer = STFTFeaturesExtractor(self.manager, f_min=self.freq_range[0], f_max=self.freq_range[1],
                                                   vmin=-35, vmax=140)

        # main layout that will contain the spectro, dataset name and options
        self.layout = QtWidgets.QHBoxLayout(self)
        self.setLayout(self.layout)
        # secondary layout (at left) that will contain dataset name and options
        self.control_layout = QtWidgets.QVBoxLayout()
        self.layout.addLayout(self.control_layout)

        # name of the dataset
        self.label = QLabel(self)
        self.label.setText(f'{manager.path.split("/")[-2]} - {manager.station_name}')  # dataset - station
        font = QtGui.QFont()
        font.setBold(True)
        self.label.setFont(font)
        self.control_layout.addWidget(self.label)

        # controller of the segment length, made of a spin box (text editor) and a slider, both interconnected
        # segment length layout
        self.segment_length_layout = QtWidgets.QGridLayout()
        self.segment_length_layout.setAlignment(Qt.AlignmentFlag.AlignBottom)
        self.control_layout.addLayout(self.segment_length_layout)

        # segment length label
        self.segment_length_label = QLabel(self)
        self.segment_length_label.setText("Segment duration (s)")
        self.segment_length_layout.addWidget(self.segment_length_label, 0, 0, 1, 1)

        # segment length spin box
        self.segment_length_doubleSpinBox = QDoubleSpinBox(self)
        self.segment_length_doubleSpinBox.setMinimum(self.MIN_SEGMENT_DURATION_S)
        self.segment_length_doubleSpinBox.setMaximum(self.MAX_SEGMENT_DURATION_S)
        self.segment_length_doubleSpinBox.setValue(2 * delta_view_s)
        self.segment_length_layout.addWidget(self.segment_length_doubleSpinBox, 1, 0, 1, 1)
        # spin box triggers to update the allowed segment date range, the slider and the plot when changed
        self.segment_length_doubleSpinBox.valueChanged.connect(self.updateDatesBounds)
        self.segment_length_doubleSpinBox.valueChanged.connect(self.updateSegmentLengthSlider)
        self.segment_length_doubleSpinBox.valueChanged.connect(self.updatePlot)

        # segment length slider
        self.segment_length_slider = QSlider(self)
        self.segment_length_slider.setMinimum(self.MIN_SEGMENT_DURATION_S)
        self.segment_length_slider.setMaximum(self.MAX_SEGMENT_DURATION_S)
        self.segment_length_slider.setOrientation(Qt.Horizontal)
        self.segment_length_layout.addWidget(self.segment_length_slider, 2, 0, 1, 3)
        # slider trigger to update the spin box when released (and recursively the plot)
        self.segment_length_slider.sliderReleased.connect(self.updateSegmentLengthEditor)

        # controller of the segment date, made of a date time editor and a slider, both interconnected
        # segment date layout
        self.segment_date_layout = QtWidgets.QGridLayout()
        self.segment_date_layout.setAlignment(Qt.AlignmentFlag.AlignBottom)
        self.control_layout.addLayout(self.segment_date_layout)

        # segment date label
        self.segment_date_label = QLabel(self)
        self.segment_date_label.setText("Segment date")
        self.segment_date_layout.addWidget(self.segment_date_label, 0, 0, 1, 1)

        # segment date time edit
        self.segment_date_dateTimeEdit = QDateTimeEdit(self)
        self.segment_date_dateTimeEdit.setInputMethodHints(Qt.ImhPreferNumbers)
        self.segment_date_dateTimeEdit.setDisplayFormat("yyyy/MM/dd hh:mm:ss")
        self.segment_date_dateTimeEdit.setDateTime(DatetimeToQdatetime(manager.dataset_start))
        self.segment_date_layout.addWidget(self.segment_date_dateTimeEdit, 1, 0, 1, 1)
        # set the initital date
        if start_date is not None:
            self.segment_date_dateTimeEdit.setDateTime(DatetimeToQdatetime(start_date))
        # date time edit triggers to update the slider and the plot when changed
        self.segment_date_dateTimeEdit.dateTimeChanged.connect(self.updateSegmentDateSlider)
        self.segment_date_dateTimeEdit.dateTimeChanged.connect(self.updatePlot)

        # segment date slider
        self.segment_date_slider = QSlider(self)
        self.segment_date_slider.setOrientation(Qt.Horizontal)
        self.segment_date_layout.addWidget(self.segment_date_slider, 2, 0, 1, 3)
        # slider trigger to update the date time edit when released (and recursively the plot)
        self.segment_date_slider.sliderReleased.connect(self.updateSegmentDateEditor)

        # mpl canvas containing the spectrogram
        self.mpl = MplCanvas(self)
        self.mpl.axes.figure.subplots_adjust(left=0.03, right=0.97, bottom=0.15, top=0.94)
        self.layout.addWidget(self.mpl)

        # timer used to prevent manual resizing of the window to cause lagging by drawing spectro too often
        self.resize_timer = QTimer()

        # initialize the different widgets
        self.layout.setStretch(1, 5)
        self.updateDatesBounds()
        self.updateSegmentLengthSlider()
        self.updateSegmentDateSlider()
        self.updatePlot()

    def updateSegmentLengthEditor(self):
        """ Update the segment length spin box after the slider is changed.
        :return: None.
        """
        self.segment_length_doubleSpinBox.setValue(self.segment_length_slider.value())

    def updateSegmentLengthSlider(self):
        """ Update the segment length slider after the spin box is changed.
        :return: None.
        """
        self.segment_length_slider.setValue(self.segment_length_doubleSpinBox.value())

    def updateSegmentDateSlider(self):
        """ Update the segment date slider after the editor is changed.
        :return: None.
        """
        self.segment_date_slider.setValue((QdatetimeToDatetime(self.segment_date_dateTimeEdit.date(),
                                                               self.segment_date_dateTimeEdit.time()) - self.manager.dataset_start).total_seconds())

    def updateSegmentDateEditor(self):
        """ Update the segment date editor after the slider is changed.
        :return: None.
        """
        delta = self.segment_length_doubleSpinBox.value() / 2
        delta = datetime.timedelta(seconds=delta)
        self.segment_date_dateTimeEdit.setDateTime(DatetimeToQdatetime(
            self.manager.dataset_start + delta + datetime.timedelta(seconds=self.segment_date_slider.value())))

    def updateDatesBounds(self):
        """ Update the current min and max allowed segment date given current segment length (to avoid getting out of
        the dataset).
        :return: None.
        """
        delta = self.segment_length_doubleSpinBox.value() / 2 + 1  # 1s of margin to avoid border conflicts
        delta = datetime.timedelta(seconds=delta)
        start = self.manager.dataset_start + delta
        end = self.manager.dataset_end - delta
        self.segment_date_dateTimeEdit.setMinimumDateTime(DatetimeToQdatetime(start))
        self.segment_date_dateTimeEdit.setMaximumDateTime(DatetimeToQdatetime(end))
        self.segment_date_slider.setMinimum(0)
        self.segment_date_slider.setMaximum((end - start).total_seconds())

    def _updateData(self):
        """ Update the spectrogram value.
        :return: None.
        """
        # in case we got here because the window has been resized, stop the timer to avoid triggering it again
        self.resize_timer.stop()
        # assign the current frequency range to the spectrogram generator
        self.STFT_computer.f_min, self.STFT_computer.f_max = self.freq_range[0], self.freq_range[1]

        segment_center = QdatetimeToDatetime(self.segment_date_dateTimeEdit.date(),
                                             self.segment_date_dateTimeEdit.time())

        delta = datetime.timedelta(seconds=self.segment_length_doubleSpinBox.value() / 2)

        # get the spectro from the extractor to be able to show it
        (f, t, spectro) = self.STFT_computer.get_features(segment_center - delta, segment_center + delta)
        # label the time axis from -delta to +delta
        extent = [min(t) - delta.total_seconds(), max(t) - delta.total_seconds(), min(f), max(f)]
        # ensure we have a fair rectangle
        w, h = self.mpl.width(), self.mpl.height()
        aspect = ((max(t) - min(t)) / (max(f) - min(f))) * (h/w)

        self.mpl.axes.cla()
        self.mpl.axes.imshow(spectro, aspect=aspect, extent=extent)
        self.mpl.axes.set_xlabel('t (s)')
        self.mpl.axes.set_ylabel('f (Hz)')

    def _draw(self):
        """ Update the plot widget.
        :return: None.
        """
        self.mpl.draw()

    def updatePlot(self):
        """ Update the spectrogram and then the plot widget.
        :return: None.
        """
        self._updateData()
        self._draw()

    def onclickGraph(self, click):
        """ Handles click operation to enable point and click move.
        :param click: Object giving details about the click.
        :return: None.
        """
        if click.xdata is not None:
            segment_center = QdatetimeToDatetime(self.segment_date_dateTimeEdit.date(),
                                                 self.segment_date_dateTimeEdit.time())
            segment_center += datetime.timedelta(seconds=click.xdata)  # move to the click location
            self.segment_date_dateTimeEdit.setDateTime(DatetimeToQdatetime(segment_center))
            self.updatePlot()  # update the spectrogram and plot it

    def onkeyGraph(self, key):
        """ Handles key press operations to enable various shortcuts.
        :param key: Object giving details about the key press.
        :return: None.
        """
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
            # play the sound with 20 times increased frequency
            data = self.manager.getSegment(segment_center - delta, segment_center + delta)
            data = data / np.max(np.abs(data))

            # write a temporary file
            to_write = np.int16(32767 * data / np.max(np.abs(data)))
            scipy.io.wavfile.write("./temp_audio.wav", int(self.manager.sampling_f * 20), to_write)
            playsound("./temp_audio.wav")
            # delete the temporary file
            os.remove('./temp_audio.wav')

        if key.key == "up":
            # increase min and max allowed frequency, respecting the limit of the Nyquist-Shannon frequency
            d = min(self.manager.sampling_f / 2 - self.freq_range[1],
                    0.1 * self.manager.sampling_f / 2)
            self.freq_range[0] += d
            self.freq_range[1] += d
            self.updatePlot()

        if key.key == "down":
            # decrase min and max allowed frequency, respecting the limit of 0
            d = min(self.freq_range[0],
                    0.1 * self.manager.sampling_f / 2)
            self.freq_range[0] -= d
            self.freq_range[1] -= d
            self.updatePlot()

        if key.key == "*":
            # decrease the maximal allowed frequency
            d = 0.05 * (self.freq_range[1] - self.freq_range[0])
            self.freq_range[1] -= d
            self.updatePlot()

        if key.key == "/":
            # increase the minimal and maximal allowed frequency, respecting the limit of 0 and the
            # Nyquist-Shannon frequency
            delta_top = min(0.05 * (self.freq_range[1] - self.freq_range[0]),
                            self.manager.sampling_f / 2 - self.freq_range[1])
            delta_bot = min(0.05 * (self.freq_range[1] - self.freq_range[0]),
                            self.freq_range[0])
            self.freq_range[0] -= delta_bot
            self.freq_range[1] += delta_top
            self.updatePlot()

        if key.key == "shift+enter":
            # put all the SpectralView widgets of the parent window to the same configuration
            segment_center = QdatetimeToDatetime(self.segment_date_dateTimeEdit.date(),
                                                 self.segment_date_dateTimeEdit.time())
            for spectralview in self.spectralViews.SpectralViews:
                spectralview.freq_range = self.freq_range.copy()
                spectralview.changeDate(segment_center)
                spectralview.changeSegmentLength(self.segment_length_doubleSpinBox.value())

        # update widgets
        self.segment_date_dateTimeEdit.setDateTime(DatetimeToQdatetime(segment_center))
        self.segment_length_doubleSpinBox.setValue(delta.total_seconds() * 2)
        self.updatePlot()

    def changeDate(self, new_date):
        """ Change the date the widget focuses on to a new one.
        :param new_date: The new datetime to focus on.
        :return: None.
        """
        # round the new date to the closest second to avoid having microseconds
        new_date = new_date + datetime.timedelta(seconds=round(new_date.microsecond / 10 ** 6))
        new_date = new_date.replace(microsecond=0)
        # simply update the segment date editor which will cascade the change
        self.segment_date_dateTimeEdit.setDateTime(DatetimeToQdatetime(new_date))

    def changeSegmentLength(self, new_length):
        """ Change the current segment length.
        :param new_length: The new length, in seconds.
        :return: None.
        """
        # simply update the segment length editor which will cascade the change
        self.segment_length_doubleSpinBox.setValue(new_length)

    def resizeEvent(self, event):
        """ Handler triggered when the window is resized. Start a timer that will update the plot to fit the window.
        The goal is to avoid updating the plot too often which would cause a great lag.
        :param event: Variable giving datails of the event.
        :return: None.
        """
        self.resize_timer.stop()
        del self.resize_timer  # ensure we destroy the timer
        self.resize_timer = QTimer()
        self.resize_timer.timeout.connect(self.updatePlot)
        # 10 ms after window resize, we redraw the plot, if no other resize happens
        self.resize_timer.start(10)


def QdatetimeToDatetime(qdate, qtime):
    """ Utility function to convert PySide dates and times to Python datetimes.
    :param qdate: A PySide date (Y,M,D).
    :param qtime: A Pyside time (h,m,s).
    :return: A Python datetime.
    """
    return datetime.datetime(qdate.year(), qdate.month(), qdate.day(),
                             qtime.hour(), qtime.minute(), qtime.second())


def DatetimeToQdatetime(datetime, day_offset=0):
    """ Utility function to convert Python datetimes to PySide datetimes.
    :param datetime: A Python datetime.
    :return: A PySide datetime.
    """
    return QDateTime(
        QDate(datetime.year, datetime.month, datetime.day + day_offset),
        QTime(datetime.hour, datetime.minute, datetime.second))
