import datetime
import sys

from PySide6.QtWidgets import (QApplication)

from GUI.windows.picker import Picker
from GUI.windows.spectral_views import SpectralViewsWindow

if __name__ == "__main__":
    # main to call in order to launch the dataset exploration tool.
    app = QApplication(sys.argv)
    app.setStyleSheet(
        "QLabel{font-size: 16pt;} QDateTimeEdit{font-size: 16pt;} QPushButton{font-size: 20pt;} QDoubleSpinBox{font-size: 16pt;}")

    database = "database.yaml"
    # todo replace by objects "events"
    to_locate = [[-40, 60, datetime.datetime(2018, 5, 10, 4, 35, 22)]]
    #window = Picker(database, to_locate)
    window = SpectralViewsWindow()
    window.show()

    sys.exit(app.exec())
