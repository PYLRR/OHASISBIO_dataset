import glob2
import yaml

from PySide6.QtWidgets import QFileDialog, QVBoxLayout, QScrollArea, QWidget
from PySide6.QtCore import (Qt)
from PySide6.QtGui import (QAction)
from PySide6.QtWidgets import (QMainWindow, QToolBar)

from GUI.widgets.spectral_view import SpectralView
from utils.data_reading.sound_file_manager import WavFilesManager, make_manager


class Picker(QMainWindow):
    def __init__(self, database_yaml, to_locate):
        super().__init__()
        self.setWindowTitle(u"T-pick")

        self.resize(1200, 800)  # size when windowed
        self.showMaximized()  # switch to full screen

        self.centralWidget = QWidget()
        # add a vertical layout to contain SpectralView widgets
        self.verticalLayout = QVBoxLayout(self.centralWidget)
        self.centralWidget.setLayout(self.verticalLayout)

        # define the central widget as a scrolling area, s.t. in case we have many spectrograms we can scroll
        self.scroll = QScrollArea(self)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.centralWidget)

        self.setCentralWidget(self.scroll)

        # list of SpectralView widgets
        self.SpectralViews = []

        self.managers = {}
        self.to_locate = to_locate
        self.initialize_managers(database_yaml)

    def add_spectral_view(self, station):
        new_SpectralView = SpectralView(self, self.managers[station])
        self.verticalLayout.addWidget(new_SpectralView)
        self.SpectralViews.append(new_SpectralView)

    def initialize_managers(self, database_yaml):
        with open(database_yaml, "r") as f:
            datasets = yaml.load(f, Loader=yaml.BaseLoader)
        for dataset in datasets:
            dt = datasets[dataset]
            for station in dt["stations"]:
                st = datasets[dataset]["stations"][station]
                if st["date_start"] != "" and st["date_end"] != "" and st["latitude"] != "" and st["latitude"] != "":
                    self.managers[station] = make_manager(f'{dt["root_dir"]}/station')
                    self.managers[station].coords = (float(st["latitude"]), float(st["longitude"]))
                    pass