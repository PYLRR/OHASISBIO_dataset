import datetime
import sys

from PySide6.QtWidgets import (QApplication)

from GUI.windows.event_picker import EventPicker
from GUI.windows.isc_matchup_checker import ISCMatchupChecker
from GUI.windows.isc_viewer import ISCViewer
from GUI.windows.spectral_viewer import SpectralViewerWindow
from utils.data_reading.catalogs.matchup_catalog import AcousticSource

if __name__ == "__main__":
    # main to call in order to launch the dataset exploration tool.
    app = QApplication(sys.argv)
    app.setStyleSheet(
        "QLabel{font-size: 16pt;} QDateTimeEdit{font-size: 16pt;} QPushButton{font-size: 20pt;} QDoubleSpinBox{font-size: 16pt;}")

    datasets_yaml = "/home/plerolland/Bureau/dataset.yaml"
    isc_file = "/home/plerolland/Bureau/catalogs/ISC/eqk_isc_2018.txt"
    velocities_file = "../../data/geo/velocities_grid.pkl"
    tissnet_checkpoint = "../../data/model_saves/TiSSNet/all/cp-0022.ckpt"
    matchups = "../../data/detections/matchups_isc_2018"
    checked_matchups = "../../data/detections/matchups_isc_2018_checked"
    window = SpectralViewerWindow(datasets_yaml)
    #window = ISCViewer(datasets_yaml, isc_file, velocities_file, tissnet_checkpoint)
    #window = ISCMatchupChecker(datasets_yaml, isc_file, velocities_file, matchups, checked_matchups, tissnet_checkpoint)
    window.show()

    sys.exit(app.exec())