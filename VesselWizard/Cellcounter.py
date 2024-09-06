import sys, os
from ClickImage import ClickImage
from ExtraWindow import ExtraWindow
from Sidebar import Sidebar
from MainTool import MainTool
from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *
from pathlib import Path
import natsort
import json
from pathlib import PurePath

# Main file of the VesselWizard program

try:
    from ctypes import windll  # Only exists on Windows, will throw an error on other systems
    myappid = 'de.dboth.cellcounter.500'
    windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
except ImportError:
    pass  


class Cellcounter(QMainWindow):

    extraWindowVisible = False

    #initialize all window elements
    def __init__(self, parent = None):
        super(Cellcounter, self).__init__(parent)
        self.setAcceptDrops(True)
        self.currentFile = False
        self.toplayout = QStackedWidget()

        #this is the initial layout allowing drag and drop
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        self.log = {}
        layout.addStretch()
        
        self.text = QLabel()
        self.text.setText("Drop picture here")
        self.text.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.text)

        self.orText = QLabel()
        self.orText.setText("or")
        self.orText.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.orText)
        
        self.button = QPushButton()
        self.button.setText("Select picture")
        self.button.clicked.connect(self.selectImage)
        layout.addWidget(self.button)

        layout.addStretch()

        widget = QWidget()
        widget.setLayout(layout)


        #this is the main layout to work on an image
        widget2 = QWidget()
        layout2 = QHBoxLayout()
        widget2.setLayout(layout2)
 
        self.clickImage = ClickImage(self)
        self.sidebar = Sidebar(self.clickImage)
        self.extraWindow = ExtraWindow(self)

        self.main = MainTool(self.clickImage, self.sidebar, self.setProgress, self.extraWindow.image)

        layout2.addWidget(self.sidebar)
        layout2.addWidget(self.clickImage)

        self.toplayout.addWidget(widget)
        self.toplayout.addWidget(widget2)

        self.toplayout.setCurrentIndex(0)
        
        self.setCentralWidget(self.toplayout)
        self.setWindowIcon(QIcon(os.path.join(os.path.dirname(__file__), 'images/icon.ico')))
        self.setWindowTitle("Cellcounter v0.5")

        toolbarTop = QToolBar("Toolbar")
        self.addToolBar(Qt.TopToolBarArea, toolbarTop)

        toolbarRight = QToolBar("Zoom Control")
        self.addToolBar(Qt.RightToolBarArea, toolbarRight)
        
        zoomin = QPushButton("+")
        toolbarRight.addWidget(zoomin)
        zoomin.clicked.connect(lambda: self.main.move(0,0,-1))

        zoomout = QPushButton("-")
        toolbarRight.addWidget(zoomout)
        zoomout.clicked.connect(lambda: self.main.move(0,0,1))

        top = QPushButton()
        top.setIcon(self.style().standardIcon(QStyle.SP_ArrowUp))
        toolbarRight.addWidget(top)
        top.clicked.connect(lambda: self.main.move(0,-1,0))
        right = QPushButton()
        right.setIcon(self.style().standardIcon(QStyle.SP_ArrowRight))
        toolbarRight.addWidget(right)
        right.clicked.connect(lambda: self.main.move(1,0,0))

        down = QPushButton()
        down.setIcon(self.style().standardIcon(QStyle.SP_ArrowDown))
        toolbarRight.addWidget(down)
        down.clicked.connect(lambda: self.main.move(0,1,0))

        

        

        left = QPushButton()
        left.setIcon(self.style().standardIcon(QStyle.SP_ArrowLeft))
        toolbarRight.addWidget(left)
        left.clicked.connect(lambda: self.main.move(-1,0,0))


        pixmapi1 = getattr(QStyle, "SP_ArrowBack")
        pixmapi2 = getattr(QStyle, "SP_ArrowForward")
        pixmapi3 = getattr(QStyle, "SP_FileDialogContentsView")

        button_action = QPushButton()
        button_action.setIcon(self.style().standardIcon(pixmapi1))
        button_action.clicked.connect(self.previousFile)
        toolbarTop.addWidget(button_action)
        
        button_action2 = QPushButton()
        button_action2.setIcon(self.style().standardIcon(pixmapi2))##
        button_action2.clicked.connect(self.nextFile)
        toolbarTop.addWidget(button_action2)
        toolbarTop.addSeparator()
        
        self.button_action3 = QPushButton()
        self.button_action3.setCheckable(True)
        self.button_action3.setIcon(self.style().standardIcon(pixmapi3))##
        self.button_action3.clicked.connect(self.toggleExtraWindow)
        toolbarTop.addWidget(self.button_action3)
        self.secondaryCountButton = QPushButton()
        self.secondaryCountButton.setCheckable(True)
        self.secondaryCountButton.setIcon(self.style().standardIcon(getattr(QStyle, "SP_DialogNoButton")))##
        self.secondaryCountButton.clicked.connect(self.setToggleButton)
        toolbarTop.addWidget(self.secondaryCountButton)
        toolbarTop.addSeparator()
        self.filelabel = QAction("...", self)
        self.filelabel.triggered.connect(self.selectImage)
        toolbarTop.addAction(self.filelabel) 

        toolbarTop.addSeparator()

        separator = QWidget()
        separator.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding);
        toolbarTop.addWidget(separator)

        self.progressbar = QProgressBar()
        self.progressbar.setTextVisible(False)
        toolbarTop.addWidget(self.progressbar)

    def setToggleButton(self):
        if self.secondaryCountButton.isChecked():
            self.secondaryCountButton.setIcon(self.style().standardIcon(getattr(QStyle, "SP_DialogYesButton")))
        else:
            self.secondaryCountButton.setIcon(self.style().standardIcon(getattr(QStyle, "SP_DialogNoButton")))


    def setProgress(self, progress):
        if progress == -1:
            self.progressbar.setMinimum(0)
            self.progressbar.setMaximum(0)
            self.progressbar.setValue(0)
            self.progressbar.setVisible(True)
        else:
            self.progressbar.setMinimum(0)
            self.progressbar.setMaximum(100)
            self.progressbar.setValue(progress)
            self.progressbar.setVisible(True)

    def selectImage(self):
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.ExistingFile)
        if dlg.exec():
            filenames = dlg.selectedFiles()
            self.fileSelection(filenames)

    def previousFile(self):
        if not self.currentFile:
            return

        directory, filename = os.path.split(self.currentFile)
        current = self.currentFile.replace('/', os.sep)
        lookup = directory
        nextPath = None
        level = 0
        while nextPath is None:
            if level > 4:
                return
            level += 1
            print("Lookup",lookup)
            images = ["*.tiff","*.tif","*.png","*.jpg","*.jpeg"]
            fileListx = []
            for image in images:
                fileListx += [str(x) for x in Path(lookup).rglob(image)]
            fileList = natsort.natsorted(fileListx, reverse=False)
            fileList = [file for file in fileList if "metadata" not in file]
            nextIndex = fileList.index(current) - 1
            if nextIndex < 0:
                lookup = Path(lookup).parent.absolute()
            else:
                nextPath = fileList[nextIndex]
        self.fileSelection([nextPath])

    def toggleExtraWindow(self):
        self.extraWindow.update()

    def nextFile(self):
        if not self.currentFile:
            return

        directory, filename = os.path.split(self.currentFile)
        current = self.currentFile.replace('/', os.sep)
        lookup = directory
        nextPath = None
        level = 0
        while nextPath is None:
            if level > 4:
                return
            level += 1
            print("Lookup",lookup)
            images = ["*.tiff","*.tif","*.png","*.jpg","*.jpeg"]
            fileListx = []
            for image in images:
                fileListx += [str(x) for x in Path(lookup).rglob(image)]
            fileList = natsort.natsorted(fileListx, reverse=False)
            fileList = [file for file in fileList if "metadata" not in file]
            nextIndex = fileList.index(current) + 1
            if nextIndex == 0 or nextIndex == len(fileList):
                lookup = Path(lookup).parent.absolute()
            else:
                nextPath = fileList[nextIndex]
        self.fileSelection([nextPath])
    
    def reset(self):
        self.button.show()
        self.orText.show()
        self.setAcceptDrops(True)
    
    def fileSelection(self,files):
        self.setAcceptDrops(True)
        self.filelabel.setText(files[0])
        self.main.setImage(files[0])
        self.showMaximized()
        self.currentFile = files[0]
        self.toplayout.setCurrentIndex(1)

    def dragLeaveEvent(self, event):
        self.reset()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            self.button.hide()
            self.orText.hide()
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        self.fileSelection(files)
    
    def sizeHint(self):
      return QSize(600,400)



def main():
   app = QApplication(sys.argv)
   ex = Cellcounter()
   ex.show()
   sys.exit(app.exec())
	
if __name__ == '__main__':
   main()