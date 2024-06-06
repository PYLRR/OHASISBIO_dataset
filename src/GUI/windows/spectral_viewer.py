import datetime

import glob2
import numpy as np
import yaml

from PySide6.QtWidgets import QFileDialog, QVBoxLayout, QScrollArea, QWidget, QInputDialog, QCheckBox, QSizePolicy, \
    QHBoxLayout, QLabel
from PySide6.QtCore import (Qt)
from PySide6.QtGui import (QAction)
from PySide6.QtWidgets import (QMainWindow, QToolBar)

from GUI.widgets.spectral_view import SpectralView, QdatetimeToDatetime
from GUI.widgets.spectral_view_tissnet import SpectralViewTissnet
from utils.data_reading.catalogs.events import AcousticEmission
from utils.data_reading.sound_data.station import Station, StationsCatalog
from utils.physics.sound_model import HomogeneousSoundModel
from utils.training.TiSSNet import TiSSNet

MIN_SPECTRAL_VIEW_HEIGHT = 400
DELTA_VIEW_S = 200
class SpectralViewerWindow(QMainWindow):
    """ Window containing several SpectralView widgets and enabling to import them one by one or in group.
    """
    def __init__(self, database_yaml, tissnet_checkpoint=None):
        """ Constructor initializing the window and setting its visual appearance.
        :param database_yaml: YAML file containing information about the available stations.
        :param tissnet_checkpoint: Checkpoint of TiSSNet model in case we want to try detections.
        """
        super().__init__()
        self.sound_model = HomogeneousSoundModel()
        self.stations = StationsCatalog(database_yaml).filter_out_undated().filter_out_unlocated()

        self.setWindowTitle(u"Acoustic viewer")

        self.resize(1200, 800)  # size when windowed
        self.showMaximized()  # switch to full screen

        self.centralWidget = QWidget()
        # add a vertical layout to contain SpectralView widgets
        self.verticalLayout = QVBoxLayout(self.centralWidget)

        self.topBarLayout = QHBoxLayout()
        self.verticalLayout.addLayout(self.topBarLayout)

        self.topBarLayout.addStretch()
        self.eventIDLabel = QLabel(self.centralWidget)
        self.eventIDLabel.setFixedSize(750, 35)
        self.topBarLayout.addWidget(self.eventIDLabel)
        self.srcEstimateLabel = QLabel(self.centralWidget)
        self.srcEstimateLabel.setFixedSize(950, 35)
        self.topBarLayout.addWidget(self.srcEstimateLabel)
        self.topBarLayout.addStretch()

        self.verticalLayout.addStretch()

        # define the central widget as a scrolling area, s.t. in case we have many spectrograms we can scroll
        self.scroll = QScrollArea(self)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.centralWidget)

        self.setCentralWidget(self.scroll)

        # add a toolbar
        self.toolBar = QToolBar(self.centralWidget)
        self.addToolBar(Qt.TopToolBarArea, self.toolBar)
        self.toolBar.actionTriggered.connect(self.toolbarAction)
        # add a button to add sound folders
        self.action_add_a_sound_folder = QAction(self)
        self.action_add_a_sound_folder_text = "Add a sound folder"
        self.action_add_a_sound_folder.setText(self.action_add_a_sound_folder_text)
        self.toolBar.addAction(self.action_add_a_sound_folder)
        # add a button to choose an event to visualize
        self.action_pick_event = QAction(self)
        self.action_pick_event_text = "View an event"
        self.action_pick_event.setText(self.action_pick_event_text)
        self.toolBar.addAction(self.action_pick_event)
        # add a button to clear the current spectral views
        self.action_clear = QAction(self)
        self.action_clear_text = "Clear"
        self.action_clear.setText(self.action_clear_text)
        self.toolBar.addAction(self.action_clear)
        # right part of the toolbar
        stretch = QWidget(self)
        stretch.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.toolBar.addWidget(stretch)
        # add a button to locate an event taking the current focuses as arrival times
        self.action_locate = QAction(self)
        self.action_locate_text = "Locate"
        self.action_locate.setText(self.action_locate_text)
        self.toolBar.addAction(self.action_locate)
        # add a checkbox to enable to broadcast keyboard commands made to spectral_view widgets
        self.toolBar.addSeparator() # to better separate the checkbox
        self.broadcast_checkbox = QCheckBox("broadcast controls")
        self.toolBar.addWidget(self.broadcast_checkbox)

        # list of SpectralView widgets
        self.SpectralViews = []

        # TiSSNet
        self.model = None
        if tissnet_checkpoint:
            self.model = TiSSNet()
            self.model.load_weights(tissnet_checkpoint)

    def _add_spectral_view(self, station, date=None):
        """ Given a station, create a spectral_view widget and add it to the window.
        :param station: The station from which to make a spectral_view.
        :param date: The date at which the spectral_view will be focused, if None it will be the dataset start.
        :return: None.
        """
        sv = SpectralView if self.model is None else SpectralViewTissnet
        new_SpectralView = sv(self, station, date=date, delta_view_s=DELTA_VIEW_S)
        self.verticalLayout.addWidget(new_SpectralView)
        self.SpectralViews.append(new_SpectralView)
        for view in self.SpectralViews:  # resize other spectral views if needed
            view.setFixedHeight(max(self.height() * 0.9 // len(self.SpectralViews), MIN_SPECTRAL_VIEW_HEIGHT))

    def _addDir(self, directory, depth=0, max_depth=3):
        """ Given a directory, add a SpectralView if it is a sound files directory, else recursively look for sound
        files.
        :param directory: The directory to consider.
        :param depth: Current depth of recursive calls in case we are in a recursive call.
        :param max_depth: Max allowed depth of recursive calls, to avoid freezing the app.
        :return: None.
        """
        if directory != "":
            files = glob2.glob(directory + "/*.wav") + glob2.glob(directory + "/*.dat") + glob2.glob(directory + "/*.w")
            # check if we have a directory of .dat or .wav files
            if len(files) > 0:
                station = None
                # check if the station is registered in the dataset
                for st in self.stations:
                    if st.path in directory:
                        station = st
                station = station or Station(directory, initialize_metadata=True)
                self._add_spectral_view(station)
            # else we recursively look for .dat or .wav files
            else:
                # check we didn't reach the max depth of recursive calls
                if depth < max_depth:
                    for subdir in glob2.glob(f'{directory}/*/'):
                        self._addDir(subdir, depth=depth+1, max_depth=max_depth)

    def onkeyGraph_spectral_view(self, spectral_view, key):
        """ Method called by a spectral_view child when a key is pressed on it.
        :param spectral_view: The spectral_view focused by the user.
        :param key: Object providing information about the key pressed.
        :return: None.
        """
        # in case the broadcast checkbox is checked, we broadcast if possible the shortcut to all spectral view
        if self.broadcast_checkbox.isChecked() and 'enter' not in key.key:
            for spectral_view in self.SpectralViews:
                spectral_view.onkeyGraph_local(key)
        else:
            spectral_view.onkeyGraph_local(key)

    def clear_spectral_views(self):
        """ Clears the current window by removing all SpectralView widgets. Also clears top bar labels.
        :return: None.
        """
        self.eventIDLabel.setText("")
        self.srcEstimateLabel.setText("")
        for spectral_view in self.SpectralViews.copy():
            self.close_spectral_view(spectral_view)

    def toolbarAction(self, qAction):
        """ Method triggered by a click on a toolbar button.
        :param qAction: Action variable passed by PySide when a toolbar button is clicked.
        :return: None.
        """
        # check which button the user clicked
        if qAction.text() == self.action_add_a_sound_folder_text:
            # open a files browser dialog to enable the user to add new SpectralView widgets to the window
            directory = QFileDialog.getExistingDirectory(self, 'directory location')  # files browser
            self._addDir(directory)  # inspection of the selected directory
        elif qAction.text() == self.action_clear_text:
            # clear spectral views
            self.clear_spectral_views()
        elif qAction.text() == self.action_pick_event_text:
            # open a selection dialog to choose an event to visualize
            with open("../data/GUI/events.yaml", "r") as f:
                events = yaml.load(f, Loader=yaml.BaseLoader)
            self.clear_spectral_views()
            to_display = list(events.keys())
            choice = QInputDialog.getItem(self, 'event to display', 'label', to_display)
            date, lat, lon = events[choice[0]]["datetime"], events[choice[0]]["lat"], events[choice[0]]["lon"]
            lat, lon = float(lat), float(lon)
            date = datetime.datetime.strptime(date, "%Y%m%d_%H%M%S")
            self.event = AcousticEmission(date, lat, lon)
            candidates = self.stations.by_date_propagation(self.event, self.sound_model)
            self.eventIDLabel.setText(f'Event selected: ({self.event.lat:.2f},{self.event.lon:.2f}) - {self.event.date}')
            for station, time_of_arrival in candidates:
                self._add_spectral_view(station, time_of_arrival)
            self.broadcast_checkbox.setChecked(True)
        elif qAction.text() == self.action_locate_text:
            self.locate()

    def close_spectral_view(self, spectral_view):
        """ Remove a particular spectral_view.
        :param spectral_view: The spectral_view to remove.
        :return: None.
        """
        spectral_view.setParent(None)
        self.SpectralViews.remove(spectral_view)

    def unfocus_spectral_view(self, spectral_view):
        """ Remove a particular spectral_view from the focus, s.t. it is not used for computations like source location.
        In case it is already unfocused, change it as focused.
        :param spectral_view: The spectral_view to remove from focus.
        :return: None.
        """
        spectral_view.focused = not spectral_view.focused

    def locate(self):
        """ Try to find the location of an event received at the focusing time on each selected spectral_view.
        :return: None.
        """
        sensors_positions = np.array([s.station.get_pos() for s in self.SpectralViews if s.focused])
        if None in sensors_positions.reshape(-1):
            self.srcEstimateLabel.setText("Station position is missing!")
            return
        detection_times = np.array([QdatetimeToDatetime(s.segment_date_dateTimeEdit.date(),
                                             s.segment_date_dateTimeEdit.time()) for s in self.SpectralViews if s.focused])
        src = self.sound_model.localize_common_source(sensors_positions, detection_times)
        time = detection_times[0] + np.mean([(detection_times[i] -
                        datetime.timedelta(seconds=self.sound_model.get_sound_travel_time(src.x[1:], sensors_positions[i]))) - detection_times[0]
                         for i in range(len(sensors_positions))])
        self.srcEstimateLabel.setText(f'Location estimate : {[float(f"{v:.2f}") for v in src.x]} at '
                                      f'{time.strftime("%Y%m%d_%H%M%S")} (cost {src.cost:.2f})')