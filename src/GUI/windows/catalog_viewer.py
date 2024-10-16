import glob2
import numpy as np
from PySide6 import QtCore

from PySide6.QtWidgets import QVBoxLayout, QScrollArea, QWidget, QHBoxLayout, QPushButton, QLabel, QTextEdit
from PySide6.QtCore import (Qt)

from GUI.windows.spectral_viewer import SpectralViewerWindow
from utils.data_reading.catalogs.catalog import merge_catalogs
from utils.data_reading.catalogs.ISC import ISC_file
from utils.data_reading.sound_data.station import StationsCatalog
from utils.physics.sound.sound_model import HomogeneousSoundModel


class CatalogViewer(SpectralViewerWindow):
    """ Class to inspect a catalog of events.
    """
    def eventFilter(self, widget, event):
        """ Internal static method to handle events like key press.
        :param widget: The widget focused when the event happened.
        :param event: The event happening.
        :return: True if anything was done, False otherwise.
        """
        if event.type() == QtCore.QEvent.KeyPress:
            if event.key() == QtCore.Qt.Key_Return:
                if widget is self.eventIndexEdit:
                    self.eventIndexEdit_enter()
                    return True
                elif widget is self.eventIDEdit:
                    self.eventIDEdit_enter()
                    return True
        return False


    def __init__(self, database_yaml, catalog_dir, velocity_profiles=None, tissnet_checkpoint=None, embedder_checkpoint=None):
        """ Initializes the window.
        :param database_yaml: Path where the list of available stations is given.
        :param catalog_dir: Directory of the catalogs of events we want to inspect.
        :param velocity_profiles: Grid of velocities for better propagation estimations, if not provided a homogeneous
        model is used.
        :param tissnet_checkpoint: Checkpoint of TiSSNet model in case we want to try detection.
        :param embedder_checkpoint: Checkpoint of embedder model in case we want to try association.
        """
        super().__init__(database_yaml, tissnet_checkpoint, embedder_checkpoint)
        self.setWindowTitle(u"Catalog viewer")

        self.resize(1200, 800)  # size when windowed
        self.showMaximized()  # switch to full screen

        self.initialize_ui()

        # list of SpectralView widgets
        self.SpectralViews = []

        # dataset reading
        self.stations = StationsCatalog(database_yaml).filter_out_undated().filter_out_unlocated()

        # catalog and current inspected ID initialization
        files = glob2.glob(f"{catalog_dir}/*.txt")
        catalogs = [ISC_file(f) for f in files]
        self.catalog = merge_catalogs(catalogs)
        self.IDs = list(self.catalog.items.keys())
        self.eventIndexLabel.setText(f"/{len(self.IDs)}")
        self.current_ID_idx = 0

        # velocity model
        if velocity_profiles is not None:
            pass
            #self.sound_model = MonthlyGridSoundModel(profiles_checkpoint=velocity_profiles,
             #                                  lat_bounds=[-72, 25], lon_bounds=[0, 180], step_paths=1)
        else:
            pass

        self.sound_model = HomogeneousSoundModel()

        self.pick_current()

    def initialize_ui(self):
        """ Initialize the UI of the window.
        :return: None.
        """
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

    def clear_spectral_views(self):
        """ Clear the spectral views window and the internal list of them.
        :return: None.
        """
        for i in reversed(range(0, self.verticalLayout.count())):
            w = self.verticalLayout.itemAt(i).widget()
            # we check w is indeed a widget because we don't want to delete a layout
            if w is not None:
                w.setParent(None)
        self.SpectralViews.clear()

    def pick_prev(self):
        """ Pick the previous event.
        :return: None
        """
        self.current_ID_idx = (self.current_ID_idx - 1) % len(self.IDs)
        self.pick_current()

    def pick_next(self):
        """ Pick the next event.
        :return: None
        """
        self.current_ID_idx = (self.current_ID_idx + 1) % len(self.IDs)
        self.pick_current()

    def eventIndexEdit_enter(self):
        """ Pick the event indexed by the user in the top panel.
        :return: None
        """
        self.current_ID_idx = int(self.eventIndexEdit.toPlainText())
        self.pick_current()

    def eventIDEdit_enter(self):
        """ Pick the event whose ID is indexed by the user in the top panel.
        :return: None
        """
        self.current_ID_idx = np.argmin(np.abs(np.array(self.IDs) - int(self.eventIDEdit.toPlainText())))
        self.pick_current()

    def is_current_idx_valid(self):
        """ Check if the current index corresponds to a valid event.
        :return: True if some stations can "hear" the event, False otherwise.
        """
        event = self.catalog[self.IDs[self.current_ID_idx]]
        stations = self.stations.by_date(event.date)
        return len(stations) > 0

    def pick_current(self):
        """ Pick the current event or the first valid one. In case none is valid, does nothing.
        :return:
        """
        n_tries = 0  # to avoid looping over all events infinitely
        while not self.is_current_idx_valid():
            n_tries += 1
            self.current_ID_idx = (self.current_ID_idx + 1) % len(self.IDs)
            if n_tries >= len(self.IDs):
                print(f"All {len(self.IDs)} events have been browsed, none can be inspected.")
                return

        event = self.catalog[self.IDs[self.current_ID_idx]]
        candidates = self.stations.by_date_propagation(event, self.sound_model)
        self.eventIndexEdit.setText(str(self.current_ID_idx))
        self.eventIDEdit.setText(str(self.IDs[self.current_ID_idx]))
        self.eventIDLabel.setText(f'({event.lat:.2f},{event.lon:.2f}) - {event.date}')
        self.clear_spectral_views()
        for station, time_of_arrival in candidates:
            self._add_spectral_view(station, time_of_arrival)