import sys

from PySide6.QtWidgets import (QApplication)
from GUI.windows.spectral_views import SpectralViewsWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet("QLabel{font-size: 16pt;} QDateTimeEdit{font-size: 16pt;} QPushButton{font-size: 20pt;} QDoubleSpinBox{font-size: 16pt;}")

    window = SpectralViewsWindow()
    window.show()

    sys.exit(app.exec())
