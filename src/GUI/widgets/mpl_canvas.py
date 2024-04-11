from PySide6.QtCore import (Qt)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

class MplCanvas(FigureCanvasQTAgg):
    """ Canvas enabling to draw a Matplotlib figure.
    """
    def __init__(self, parent=None, fig_size=None):
        """ Constructor initializing the canvas.
        :param parent: Parent of the canvas.
        """
        if fig_size:
            fig = Figure()
        else:
            fig = Figure()
        super(MplCanvas, self).__init__(fig)
        self.parent = parent
        self.axes = fig.add_subplot(111)

        def onclick(click):
            """ Redirect click event to the parent.
            :param click: Variable giving information about the click.
            :return: None.
            """
            if parent:
                self.parent.onclickGraph(click)

        def onkey(key):
            """ Redirect key press event to the parent.
            :param key: Variable giving information about the key press.
            :return: None.
            """
            if parent:
                self.parent.onkeyGraph(key)

        # assign the triggers
        fig.canvas.mpl_connect('button_press_event', onclick)
        fig.canvas.mpl_connect('key_press_event', onkey)
        fig.canvas.mpl_connect("scroll_event", self.scroll)
        self.setFocusPolicy(Qt.StrongFocus)  # allow focusing on the widget

    def scroll(self, e):
        val = self.parent.spectralViewer.scroll.verticalScrollBar().value()
        self.parent.spectralViewer.scroll.verticalScrollBar().setValue(val - 50*e.step)
