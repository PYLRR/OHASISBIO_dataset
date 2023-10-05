from PySide6.QtCore import (Qt)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.parent = parent
        self.axes = fig.add_subplot(111)

        def onclick(click):
            self.parent.onclickGraph(click)

        def onkey(key):
            self.parent.onkeyGraph(key)
        def onscroll(scroll_event):
            self.parent.onscroll(scroll_event)

        fig.canvas.mpl_connect('button_press_event', onclick)
        fig.canvas.mpl_connect('key_press_event', onkey)
        fig.canvas.mpl_connect('scroll_event', onscroll)
        super(MplCanvas, self).__init__(fig)
        self.setFocusPolicy(Qt.StrongFocus)