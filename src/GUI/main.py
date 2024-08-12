import sys
from PySide6.QtWidgets import QApplication

from GUI.windows.catalog_viewer import CatalogViewer
from GUI.windows.main_menu import MainMenuWindow
from GUI.windows.spectral_viewer import SpectralViewerWindow
from utils.data_reading.catalogs.isc import ISC_file
from utils.data_reading.catalogs.ross import Ross_file

if __name__ == "__main__":
    # main to call in order to launch the dataset exploration tool.
    app = QApplication(sys.argv)
    app.setStyleSheet(
        "QLabel{font-size: 16pt;} QDateTimeEdit{font-size: 16pt;} QPushButton{font-size: 20pt;} QDoubleSpinBox{font-size: 16pt;}")


    datasets_yaml = "/home/plerolland/Bureau/dataset.yaml"
    isc_file = "/home/plerolland/Bureau/catalogs/ISC/eqk_isc_revbull_2011.txt"
    ross_file = "/home/plerolland/Téléchargements/temp/EA_CTBTO_catalog_all.dat"
    tissnet_checkpoint = "../data/model_saves/TiSSNet/all/cp-0022.ckpt"

    # Menu choices
    menu_choices = [("Spectral Viewer", lambda: SpectralViewerWindow(datasets_yaml, tissnet_checkpoint=tissnet_checkpoint)),
                    ("ISC Viewer", lambda: CatalogViewer(datasets_yaml, ISC_file(isc_file), tissnet_checkpoint=tissnet_checkpoint)),
                    ("Ross Viewer", lambda: CatalogViewer(datasets_yaml, Ross_file(ross_file), tissnet_checkpoint=tissnet_checkpoint))]

    window = MainMenuWindow(menu_choices)
    window.show()

    sys.exit(app.exec())

