from PySide6.QtCore import (Qt)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

class MplCanvas(FigureCanvasQTAgg):
    """ Canvas enabling to draw a Matplotlib figure.
    """
    def __init__(self, parent=None):
        """ Constructor initializing the canvas.
        :param parent: Parent of the canvas.
        """
        fig = Figure()
        super(MplCanvas, self).__init__(fig)
        self.parent = parent
        self.axes = fig.add_subplot(111)

        def onclick(click):
            """ Redirect click event to the parent.
            :param click: Variable giving information about the click.
            :return: None.
            """
            self.parent.onclickGraph(click)

        def onkey(key):
            """ Redirect key press event to the parent.
            :param key: Variable giving information about the key press.
            :return: None.
            """
            self.parent.onkeyGraph(key)

        # assign the triggers
        fig.canvas.mpl_connect('button_press_event', onclick)
        fig.canvas.mpl_connect('key_press_event', onkey)
        self.setFocusPolicy(Qt.StrongFocus)  # allow focusing on the widget
