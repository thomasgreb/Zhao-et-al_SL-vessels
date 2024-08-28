from PySide6.QtCore import *
import uuid, os, json, qimage2ndarray, cv2
from PySide6.QtGui import *
from PySide6.QtWidgets import *

#A resizeable QT image widget with additional handling for clicks
class ClickImage(QWidget):
    pixmap = False
    _sizeHint = QSize()
    ratio = Qt.KeepAspectRatio
    clickEvent = lambda self,x: x
    main = None
    transformation = Qt.SmoothTransformation
    v = {"zoom": 1, "x": 0, "y": 0}

    def __init__(self, cc, pixmap=None):
        super().__init__()
        self.counters = {}
        self.cellCounter = cc
        pal = self.palette()
        self.setContentsMargins(0, 0, 0, 0)
        pal.setColor(QPalette.Window, Qt.black)
        self.setAutoFillBackground(True)
        self.setPalette(pal)
        self.setStyleSheet("border: 1px solid #aaa; background-color: black")
        self.setPixmap(pixmap)

    def setMain(self, main):
        self.main = main

    def setImage(self, image):
        mat = cv2.cvtColor(cv2.imread(image),cv2.COLOR_BGR2RGB)
        self.setMat(mat)

    def setMat(self, mat):
        mat = mat[self.v["y"] : self.v["y"] + round(self.v["zoom"]*mat.shape[0]), self.v["x"] : self.v["x"] + round(self.v["zoom"]*mat.shape[1])]
        self.setPixmap(QPixmap.fromImage(qimage2ndarray.array2qimage(mat)))

    def setPixmap(self, pixmap):
        if self.pixmap != pixmap:
            self.pixmap = pixmap
            if isinstance(pixmap, QPixmap):
                self._sizeHint = pixmap.size()
            else:
                self._sizeHint = QSize()
            self.updateGeometry()
            self.updateScaled()

    def setAspectRatio(self, ratio):
        if self.ratio != ratio:
            self.ratio = ratio
            self.updateScaled()

    def setTransformation(self, transformation):
        if self.transformation != transformation:
            self.transformation = transformation
            self.updateScaled()

    def updateScaled(self):
        if self.pixmap:
            self.scaled = self.pixmap.scaled(
                self.size(), self.ratio, self.transformation
            )
        self.update()

    def sizeHint(self):
        return self._sizeHint

    def resizeEvent(self, event):
        self.updateScaled()

    def paintEvent(self, event):
        if not self.pixmap:
            return
        qp = QPainter(self)
        r = self.scaled.rect()
        r.moveCenter(self.rect().center())
        qp.drawPixmap(r, self.scaled)
        for counter in self.counters:
            self.counters[counter].refreshPosition()

    def registerClickEvent(self, fn):
        self.clickEvent = fn

    def setCurrentView(self, v):
        self.v = v

    def mousePressEvent(self, event):
        if self.main is None:
            pass
        pos = event.pos()
        parentSize = self.size()
        imageSize = self.pixmap.size()
        imageAspectRatio = imageSize.width() / imageSize.height()
        if (parentSize.width() / parentSize.height()) >= imageAspectRatio:
            realImageHeight = parentSize.height()
            realImageWidth = realImageHeight * imageAspectRatio
            imageX = (parentSize.width() - realImageWidth) / 2
            imageY = 0
        else:
            realImageWidth = parentSize.width()
            realImageHeight = realImageWidth / imageAspectRatio
            imageX = 0
            imageY = (parentSize.height() - realImageHeight) / 2
        ox = max(0, min(1, (pos.x() - imageX) / realImageWidth))
        oy = max(0, min(1, (pos.y() - imageY) / realImageHeight))
        x = (self.v["x"] + ox*self.main.settings["size"]["x"]*self.v["zoom"]) / self.main.settings["size"]["x"]
        y = (self.v["y"] + oy*self.main.settings["size"]["y"]*self.v["zoom"]) / self.main.settings["size"]["y"]
        self.clickEvent((x, y, self.cellCounter.secondaryCountButton.isChecked() if self.cellCounter is not None else False))
