from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QPushButton


class MainMenuWindow(QMainWindow):
    """ Generic menu offering a vertical layout of choices, each linked with a triggerred function.
    """
    def __init__(self, menu_choices):
        """ Initializes the menu.
        :param menu_choices: A list of tuples (text_to_display, function_triggerred_when_clicked) representing choices.
        """
        super(MainMenuWindow, self).__init__()
        self.setWindowTitle(u"Main menu")

        self.resize(400, 200)
        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)
        self.verticalLayout = QVBoxLayout(self.centralWidget)
        for choice_name, window in menu_choices:
            button = QPushButton(self.centralWidget)
            button.setText(choice_name)
            button.setFixedHeight(40)
            self.verticalLayout.addWidget(button)
            button.clicked.connect(self.display_window(window))

    def display_window(self, window):
        """ Method to open a window and record it as an attribute.
        :param window: Window to open.
        :return: A function that opens the window.
        """
        def f():
            self.window = window()
            self.window.show()
        return f