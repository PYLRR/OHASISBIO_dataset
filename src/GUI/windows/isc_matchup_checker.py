import datetime
import os.path
import pickle

import glob2
import yaml
from PySide6 import QtGui, QtCore

from PySide6.QtWidgets import QVBoxLayout, QScrollArea, QWidget, QHBoxLayout, QPushButton, QLabel, QTextEdit
from PySide6.QtCore import (Qt, SIGNAL, Signal)
from PySide6.QtWidgets import (QMainWindow)

from GUI.widgets.spectral_view import SpectralView, QdatetimeToDatetime
from GUI.widgets.spectral_view_matchup_check import SpectralViewMatchupCheck
from GUI.widgets.spectral_view_tissnet import SpectralViewTissnet
from GUI.windows.isc_viewer import ISCViewer, DELTA_VIEW_S
from utils.data_reading.catalogs.isc import ISC_file
from utils.data_reading.catalogs.matchup_catalog import AcousticDetection, Matchup
from utils.data_reading.sound_data.sound_file_manager import make_manager
from utils.data_reading.sound_data.station import StationsCatalog
from utils.physics.sound_model import MonthlyGridSoundModel, HomogeneousSoundModel
from utils.training.TiSSNet import TiSSNet

MIN_MATCHUP_SIZE = 3
class ISCMatchupChecker(ISCViewer):
    def __init__(self, database_yaml, isc_file, velocity_profiles, matchups, checked_matchups, tissnet_checkpoint=None):
        with open(matchups, "rb") as f:
            self.matchups = pickle.load(f)
        self.checked_matchups_file = checked_matchups
        self.checked_matchups = {}
        checked_matchups_files = sorted(glob2.glob(f'{checked_matchups}*'))
        if len(checked_matchups_files) > 0:
            with open(checked_matchups_files[-1], "rb") as f:
                self.checked_matchups = pickle.load(f)
            print(f"{len(list(self.checked_matchups.keys()))} previous checks loaded.")
        new_matchups = {}
        for ID, matchup in self.matchups.items():
            if len(matchup.detections) >= MIN_MATCHUP_SIZE:
                new_matchups[ID] = matchup
        self.matchups = new_matchups
        super().__init__(database_yaml, isc_file, velocity_profiles, tissnet_checkpoint)
        self.scroll.installEventFilter(self)


    def eventFilter(self, widget, event):
        if super().eventFilter(widget, event):
            return True
        if event.type() == QtCore.QEvent.KeyPress:
            if not widget is self.eventIndexEdit and not widget is self.eventIDEdit:
                if event.key() == QtCore.Qt.Key_Return:
                    self.validate()
                    return True
                if event.key() == QtCore.Qt.Key_Backspace:
                    self.invalidate()
                    return True
        return False


    def initialize_isc(self, isc_file):
        self.isc = ISC_file(isc_file)
        self.IDs = list(self.matchups.keys())
        self.eventIndexLabel.setText(f"/{len(self.IDs)}")
        self.current_ID_idx = 0

    def add_spectral_view(self, station, date):
        station.load_data()
        new_SpectralView = SpectralViewMatchupCheck(self, station, start_date=date, delta_view_s=DELTA_VIEW_S)
        self.scrollVerticalLayout.addWidget(new_SpectralView)
        self.SpectralViews.append(new_SpectralView)

        stations_in_matchup = [d.station for d in self.matchups[self.IDs[self.current_ID_idx]].detections]
        if station in stations_in_matchup:
            detection_idx = stations_in_matchup.index(station)
            detection = self.matchups[self.IDs[self.current_ID_idx]].detections[detection_idx]
            offset = (detection.date - date).total_seconds()
            new_SpectralView.setPointerPos(offset)

    def is_current_idx_valid(self):
        event = self.isc[self.IDs[self.current_ID_idx]]
        stations = self.stations.by_date(event.date)
        return len(stations) > 0 and self.IDs[self.current_ID_idx] not in self.checked_matchups.keys()

    def validate(self):
        val = []
        for i, spectralView in enumerate(self.SpectralViews):
            if spectralView.pointer_pos:
                segment_center = QdatetimeToDatetime(spectralView.segment_date_dateTimeEdit.date(),
                                                     spectralView.segment_date_dateTimeEdit.time())
                date = segment_center + datetime.timedelta(seconds=spectralView.pointer_pos)
                val.append(AcousticDetection(spectralView.station.light_copy(), date))
        val = Matchup(val)
        self.checked_matchups[self.IDs[self.current_ID_idx]] = val
        self.save()
        self.pick_next()

    def invalidate(self):
        self.checked_matchups[self.IDs[self.current_ID_idx]] = Matchup([])
        self.save()
        self.pick_next()

    def save(self):
        with open(f'{self.checked_matchups_file}_{datetime.datetime.now().strftime("%m%d_%H%M%S")}', "wb") as f:
            pickle.dump(self.checked_matchups, f)