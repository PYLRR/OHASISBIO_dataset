import glob

from PySide6.QtWidgets import QFileDialog, QVBoxLayout, QScrollArea
from PySide6.QtCore import (QMetaObject, Qt)
from PySide6.QtGui import (QAction)
from PySide6.QtWidgets import (QMainWindow, QToolBar, QWidget)

from GUI.widgets.spectral_view import SpectralView
from utils.data_reading.sound_file_manager import WavFilesManager

"""
    Custom Qt Widget to display several SpectralView
"""
class Ui_SpectralViews(object):
    def setupUi(self, SpectralViews):
        if not SpectralViews.objectName():
            SpectralViews.setObjectName(u"GUI")

        SpectralViews.resize(1200, 800)
        SpectralViews.showMaximized()

        self.actionAdd_a_wav_folder = QAction(SpectralViews)

        self.centralwidget = QWidget(SpectralViews)
        self.centralwidget.setObjectName(u"centralwidget")
        self.scroll = QScrollArea()
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.centralwidget)
        SpectralViews.setCentralWidget(self.scroll)

        self.verticalLayout = QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName(u"verticalLayout")

        self.toolBar = QToolBar(SpectralViews)
        self.toolBar.setObjectName(u"toolBar")
        SpectralViews.addToolBar(Qt.TopToolBarArea, self.toolBar)

        self.toolBar.addAction(self.actionAdd_a_wav_folder)

        self.toolBar.actionTriggered.connect(SpectralViews.addDir)
        SpectralViews.setWindowTitle(u"Acoustic viewer")
        self.actionAdd_a_wav_folder.setText("Add a sound folder")
        self.toolBar.setWindowTitle(u"toolBar")

        QMetaObject.connectSlotsByName(SpectralViews)
        self.SpectralViews = []

    def addManager(self, manager):
        new_SpectralView = SpectralView(manager, self)
        self.verticalLayout.addWidget(new_SpectralView)
        self.SpectralViews.append(new_SpectralView)


class SpectralViewsWindow(QMainWindow):
    def __init__(self):
        super(SpectralViewsWindow, self).__init__()
        self.ui = Ui_SpectralViews()
        self.ui.setupUi(self)

        self.wav_folders = []
        self.managers = []

    def _addDir(self, directory):
        if directory != "":
            files = glob.glob(directory + "/*.wav")
            if len(files) > 0:
                # we have a directory of .wav files
                self.wav_folders.append(directory)
                self.ui.addManager(WavFilesManager(directory))
            else:
                # we found no sound file and look recursively for sound files in subfolders
                for subdir in glob.glob(f'{directory}/**/'):
                    self._addDir(subdir)

    def addDir(self, qAction):
        if qAction.text() == u"Add a sound folder":
            directory = QFileDialog.getExistingDirectory(self, 'directory location')
            self._addDir(directory)