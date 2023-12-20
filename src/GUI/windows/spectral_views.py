import glob2

from PySide6.QtWidgets import QFileDialog, QVBoxLayout, QScrollArea, QWidget
from PySide6.QtCore import (Qt)
from PySide6.QtGui import (QAction)
from PySide6.QtWidgets import (QMainWindow, QToolBar)

from GUI.widgets.spectral_view import SpectralView
from utils.data_reading.sound_file_manager import WavFilesManager

MIN_SPECTRAL_VIEW_HEIGHT = 400
class SpectralViewsWindow(QMainWindow):
    """ Window containing several SpectralView widgets and enabling to import them one by one or in group.
    """
    def __init__(self):
        """ Constructor initializing the window and setting its visual appearance.
        """
        super().__init__()
        self.setWindowTitle(u"Acoustic viewer")

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


        # add a toolbar with a button to add sound folders
        self.toolBar = QToolBar(self.centralWidget)
        self.addToolBar(Qt.TopToolBarArea, self.toolBar)
        self.actionAdd_a_wav_folder = QAction(self)
        self.actionAdd_a_wav_folder_text = "Add a sound folder"
        self.actionAdd_a_wav_folder.setText(self.actionAdd_a_wav_folder_text)
        self.toolBar.addAction(self.actionAdd_a_wav_folder)
        self.toolBar.actionTriggered.connect(self.addDir)

        # list of SpectralView widgets
        self.SpectralViews = []

    def _addDir(self, directory, depth=0, max_depth=3):
        """ Given a directory, add a SpectralView if it is a sound files directory, else recursively look for sound
        files.
        :param directory: The directory to consider.
        :param depth: Current depth of recursive calls in case we are in a recursive call.
        :param max_depth: Max allowed depth of recursive calls, to avoid freezing the app.
        :return: None.
        """
        if directory != "":
            files = glob2.glob(directory + "/*.wav")
            # check if w have a directory of .wav files
            if len(files) > 0:
                new_SpectralView = SpectralView(self, WavFilesManager(directory))
                self.verticalLayout.addWidget(new_SpectralView)
                self.SpectralViews.append(new_SpectralView)
                for view in self.SpectralViews:
                    view.setFixedHeight(max(self.height()*0.9//len(self.SpectralViews), MIN_SPECTRAL_VIEW_HEIGHT))
            # else we recursively look for .wav files
            else:
                # check we didn't reach the max depth of recursive calls
                if depth < max_depth:
                    for subdir in glob2.glob(f'{directory}/*/'):
                        self._addDir(subdir, depth=depth+1, max_depth=max_depth)

    def addDir(self, qAction):
        """ Open a files browser dialog to enable the user to add new SpectralView widgets to the window.
        :param qAction: Action variable passed by PySide when a toolbar button is clicked.
        :return: None.
        """
        # check the user clicked the add directory button
        if qAction.text() == self.actionAdd_a_wav_folder_text:
            directory = QFileDialog.getExistingDirectory(self, 'directory location')  # files browser
            self._addDir(directory)  # inspection of the selected directory
