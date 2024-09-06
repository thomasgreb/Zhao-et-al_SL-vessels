from PySide6.QtCore import *
from ClickImage import ClickImage
import os
from PySide6.QtGui import *
from PySide6.QtWidgets import *

#Extra window showing the original image
class ExtraWindow(QWidget):


    def __init__(self, main):
        super().__init__()
        self.main = main
        self.image = ClickImage(None)
        xl = QVBoxLayout()
        xl.addWidget(self.image)
        self.setLayout(xl)
        self.setWindowIcon(QIcon(os.path.join(os.path.dirname(__file__), 'images/icon.ico')))
        self.setWindowTitle("Original image")
    

    def update(self):
        state = self.main.button_action3.isChecked()
        print(state)
        if state:
            self.show()
        else:
            self.hide()
    
    def closeEvent(self, event):
        self.main.button_action3.setChecked(False)
        event.accept()

        
    