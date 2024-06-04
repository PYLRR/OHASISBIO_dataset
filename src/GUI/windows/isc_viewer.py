import datetime

import numpy as np
import yaml
from PySide6 import QtGui, QtCore

from PySide6.QtWidgets import QVBoxLayout, QScrollArea, QWidget, QHBoxLayout, QPushButton, QLabel, QTextEdit
from PySide6.QtCore import (Qt, SIGNAL, Signal)
from PySide6.QtWidgets import (QMainWindow)

from GUI.widgets.spectral_view import SpectralView
from GUI.widgets.spectral_view_tissnet import SpectralViewTissnet
from GUI.windows.spectral_viewer import SpectralViewerWindow
from utils.data_reading.catalogs.isc import ISC_file
from utils.data_reading.sound_data.sound_file_manager import make_manager
from utils.data_reading.sound_data.station import StationsCatalog
from utils.physics.sound_model import MonthlyGridSoundModel, HomogeneousSoundModel
from utils.training.TiSSNet import TiSSNet

DELTA_VIEW_S = 200

class ISCViewer(SpectralViewerWindow):
    def eventFilter(self, widget, event):
        if event.type() == QtCore.QEvent.KeyPress:
            if event.key() == QtCore.Qt.Key_Return:
                if widget is self.eventIndexEdit:
                    self.eventIndexEdit_enter()
                    return True
                elif widget is self.eventIDEdit:
                    self.eventIDEdit_enter()
                    return True
        return False

    def __init__(self, database_yaml, isc_file, velocity_profiles, tissnet_checkpoint=None):
        super().__init__(database_yaml)
        self.setWindowTitle(u"T-pick")

        self.resize(1200, 800)  # size when windowed
        self.showMaximized()  # switch to full screen

        self.initialize_ui()

        # list of SpectralView widgets
        self.SpectralViews = []

        self.stations = StationsCatalog(database_yaml).filter_out_undated().filter_out_unlocated()

        self.initialize_isc(isc_file)

        self.sound_model = MonthlyGridSoundModel(profiles_checkpoint=velocity_profiles,
                                           lat_bounds=[-72, 25], lon_bounds=[0, 180], step_paths=1)
        self.sound_model = HomogeneousSoundModel()

        self.model = None
        if tissnet_checkpoint:
            self.model = TiSSNet()
            self.model.load_weights(tissnet_checkpoint)

        self.current_ID_idx = 0
        self.pick_current(force=True)

        self.pick_current()

    def initialize_ui(self):
        self.centralWidget = QWidget()

        self.verticalLayout = QVBoxLayout(self.centralWidget)

        self.topBarLayout = QHBoxLayout()
        self.verticalLayout.addLayout(self.topBarLayout)

        self.topBarLayout.addStretch()
        self.previousButton = QPushButton(self.centralWidget)
        self.previousButton.setText(u"Prev")
        self.previousButton.setStyleSheet("background-color: gray")
        self.previousButton.setFixedHeight(40)
        self.topBarLayout.addWidget(self.previousButton)
        self.previousButton.clicked.connect(self.pick_prev)

        self.eventLayout = QVBoxLayout()
        self.topBarLayout.addLayout(self.eventLayout)
        self.eventIndexEditLayout = QHBoxLayout()
        self.eventIDEditLayout = QHBoxLayout()
        self.eventLayout.addLayout(self.eventIndexEditLayout)
        self.eventLayout.addLayout(self.eventIDEditLayout)

        self.eventIndexEditLayout.addStretch()
        self.eventIndexEdit = QTextEdit(self.centralWidget)
        self.eventIndexEdit.setFixedSize(100, 35)
        self.eventIndexEdit.setFontPointSize(16)
        self.eventIndexEdit.installEventFilter(self)
        self.eventIndexEditLayout.addWidget(self.eventIndexEdit)
        self.eventIndexLabel = QLabel(self.centralWidget)
        self.eventIndexLabel.setFixedSize(100, 35)
        self.eventIndexEditLayout.addWidget(self.eventIndexLabel)
        self.eventIndexEditLayout.addStretch()

        self.eventIDEditLayout.addStretch()
        self.eventIDEdit = QTextEdit(self.centralWidget)
        self.eventIDEdit.setFixedSize(150, 35)
        self.eventIDEdit.setFontPointSize(16)
        self.eventIDEdit.installEventFilter(self)
        self.eventIDEditLayout.addWidget(self.eventIDEdit)
        self.eventIDLabel = QLabel(self.centralWidget)
        self.eventIDLabel.setFixedSize(450, 35)
        self.eventIDEditLayout.addWidget(self.eventIDLabel)
        self.eventIDEditLayout.addStretch()

        self.nextButton = QPushButton(self.centralWidget)
        self.nextButton.setText(u"Next")
        self.nextButton.setStyleSheet("background-color: gray")
        self.nextButton.setFixedHeight(40)
        self.topBarLayout.addWidget(self.nextButton)
        self.nextButton.clicked.connect(self.pick_next)

        self.srcEstimateLabel = QLabel(self.centralWidget)
        self.srcEstimateLabel.setFixedSize(750, 35)
        self.topBarLayout.addWidget(self.srcEstimateLabel)
        self.topBarLayout.addStretch()

        # define the central widget as a scrolling area, s.t. in case we have many spectrograms we can scroll
        self.scroll = QScrollArea(self)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.centralWidget)

        self.setCentralWidget(self.scroll)

        # add a vertical layout to contain SpectralView widgets
        self.scrollVerticalLayout = QVBoxLayout()
        self.verticalLayout.addLayout(self.scrollVerticalLayout)

    def initialize_isc(self, isc_file):
        self.isc = ISC_file(isc_file)

        to_del = set()
        for e in to_del:
            del self.isc.items[e]


        self.IDs = list(self.isc.items.keys())
        self.eventIndexLabel.setText(f"/{len(self.IDs)}")
        self.current_ID_idx = 0

    def add_spectral_view(self, station, date):
        station.load_data()
        new_SpectralView = SpectralViewTissnet(self, station, start_date=date, delta_view_s=DELTA_VIEW_S)
        self.scrollVerticalLayout.addWidget(new_SpectralView)
        self.SpectralViews.append(new_SpectralView)

    def clear_spectral_views(self):
        for i in reversed(range(self.scrollVerticalLayout.count())):
            self.scrollVerticalLayout.itemAt(i).widget().setParent(None)
        self.SpectralViews.clear()

    def pick_prev(self):
        self.current_ID_idx = (self.current_ID_idx - 1) % len(self.IDs)
        self.pick_current()

    def pick_next(self):
        self.current_ID_idx = (self.current_ID_idx + 1) % len(self.IDs)
        self.pick_current()

    def eventIndexEdit_enter(self):
        self.current_ID_idx = int(self.eventIndexEdit.toPlainText())
        self.pick_current(force=True)

    def eventIDEdit_enter(self):
        self.current_ID_idx = np.argmin(np.abs(np.array(self.IDs) - int(self.eventIDEdit.toPlainText())))
        self.pick_current(force=True)

    def is_current_idx_valid(self):
        event = self.isc[self.IDs[self.current_ID_idx]]
        stations = self.stations.by_date(event.date)
        return len(stations) > 0

    def pick_current(self, force=False):
        if not force:
            while not self.is_current_idx_valid():
                self.current_ID_idx = (self.current_ID_idx + 1) % len(self.IDs)

        event = self.isc[self.IDs[self.current_ID_idx]]
        candidates = self.stations.by_date_propagation(event, self.sound_model)
        self.eventIndexEdit.setText(str(self.current_ID_idx))
        self.eventIDEdit.setText(str(self.IDs[self.current_ID_idx]))
        self.eventIDLabel.setText(f'({event.lat:.2f},{event.lon:.2f}) - {event.date}')
        self.clear_spectral_views()
        for station, time_of_arrival in candidates:
            self.add_spectral_view(station, time_of_arrival)