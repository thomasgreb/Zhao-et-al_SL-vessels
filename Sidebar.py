from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *
from functools import partial
import math
from debounce import debounce


class Sidebar(QTabWidget):
    def __init__(self, clickImage):
        super().__init__()
        self.widgets = {}
        self.clickImage = clickImage
        layout = QVBoxLayout()
        self.setContentsMargins(0, 0, 0, 0)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        self.setMinimumWidth(300)
        self.setTabPosition(QTabWidget.West)
        self.currentChanged.connect(self.onCurrentChange)

    def createCorrection(self):
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        manual = self.makeGroupbox("Manual correction", layout)
        manual.addWidget(self.makeSlider("Alpha", "correction_alpha", 100, 0, 10))
        manual.addWidget(self.makeSlider("Beta", "correction_beta", 100, -100, 100))
        manual.addWidget(self.makeButton("Manual correction", "image_correction"))
        auto = self.makeGroupbox("Autocorrect", layout)
        auto.addWidget(self.makeButton("Autocorrect", "autocorrect"))
        layout.addStretch()
        return widget

    def createDetection(self):
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        detection = self.makeGroupbox("Cell detection", layout)
        detection.addWidget(
            self.makeSlider("Threshold", "detection_binary", factor=1, min=0, max=255)
        )
        detection.addWidget(
            self.makeSlider(
                "Lower Size Cutoff", "detection_lower", factor=10, min=0.1, max=99.9
            )
        )
        detection.addWidget(
            self.makeSlider(
                "Upper Size Cutoff", "detection_higher", factor=10, min=0.1, max=99.9
            )
        )
        detection.addWidget(self.makeButton("Detect cells", "detect_cells"))

        layout.addStretch()
        return widget

    def produceError(self, message):
        dlg = QMessageBox(self)
        dlg.setWindowTitle("Error")
        dlg.setText(message)
        dlg.exec_()

    def createStaining(self):
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        staining = self.makeGroupbox("Color thresholds", layout)
        staining.addWidget(
            self.makeSlider(
                "Lower Blue Threshold", "staining_lowerH", factor=0.5, min=0, max=255
            )
        )
        staining.addWidget(
            self.makeSlider(
                "Upper Blue Threshold", "staining_upperH", factor=0.5, min=0, max=255
            )
        )
        staining.addWidget(
            self.makeSlider(
                "Minimum Saturation", "staining_saturation", factor=0.5, min=0, max=255
            )
        )
        staining.addWidget(
            self.makeSlider(
                "Minimum Brightness", "staining_brightness", factor=0.5, min=0, max=255
            )
        )
        staining.addWidget(
            self.makeSlider("Erosion", "staining_erosion", factor=1, min=0, max=6)
        )
        staining.addWidget(
            self.makeSlider("Dilation", "staining_dilation", factor=1, min=0, max=6)
        )
        staining.addWidget(self.makeButton("Detect staining", "detect_staining"))
        layout.addStretch()
        return widget

    def createROI(self):
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        search = self.makeGroupbox("Automatic cambium search", layout)
        search.addWidget(
            self.makeSlider("Erosion", "roi_erosion", factor=1, min=0, max=6)
        )
        search.addWidget(
            self.makeSlider("Blur", "roi_blur", factor=0.1, min=0, max=100)
        )
        search.addWidget(
            self.makeSlider("Cell threshold", "roi_threshold", factor=1, min=0, max=255)
        )
        search.addWidget(
            self.makeSlider("Cambium cutoff", "roi_cutoff", factor=500, min=0, max=0.4)
        )
        search.addWidget(self.makeButton("Auto detect Cambium", "detect_cambium"))
        manual = self.makeGroupbox("Manual cambium selection", layout)
        startManualCambium = QPushButton("Start")
        startManualCambium.clicked.connect(self.startManualCambiumSelection)
        manual.addWidget(startManualCambium)
        layout.addStretch()
        return widget

    def createCounting(self):
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        search = self.makeGroupbox("Cell counting", layout)
        search.addWidget(
            self.makeSlider("Dilation", "counting_dilation", factor=1, min=0, max=6)
        )
        search.addWidget(
            self.makeSlider("Threshold", "counting_threshold", factor=20, min=0, max=1)
        )
        search.addWidget(
            self.makeSlider(
                "Lower Size Cutoff", "counting_lower", factor=10, min=0.1, max=99.9
            )
        )
        search.addWidget(
            self.makeSlider(
                "Upper Size Cutoff", "counting_higher", factor=10, min=0.1, max=99.9
            )
        )
        search.addWidget(
            self.makeSlider(
                "Minimum Circularity", "counting_minCircularity", factor=1, min=0, max=100
            )
        )
        search.addWidget(self.makeButton("Auto detect xylem", "detect_xylem"))
        manual = self.makeGroupbox("Analysis", layout)
        self.analysis = QLabel()
        manual.addWidget(self.analysis)
        manual.addWidget(self.makeButton("Analyze and save", "analyze"))
        layout.addStretch()
        return widget

    def makeGroupbox(self, name, parent):
        manual = QGroupBox(name)
        mlayout = QVBoxLayout()
        manual.setLayout(mlayout)
        parent.addWidget(manual)
        return mlayout

    def value_changed(self, id, value):
        widget = self.widgets[id]
        if widget["type"] == "slider":
            realValue = widget["widget"].value() / widget["factor"]
            widget["label"].setText(
                str(round(realValue, math.ceil(math.log(widget["factor"], 10))))
            )
        self.propagateValue(id, realValue)

    def startManualCambiumSelection(self):
        dlg = QMessageBox(self)
        dlg.setWindowTitle("Step 1")
        dlg.setText("Please click on the center of the hypocotyl.")
        dlg.exec_()
        self.clickImage.registerClickEvent(self.nextStepCambium)

    def nextStepCambium(self, position):
        dlg = QMessageBox(self)
        dlg.setWindowTitle("Step 2")
        dlg.setText("Please click on a point on the cambium.")
        dlg.exec_()
        self.clickImage.registerClickEvent(partial(self.lastStepCambium, position))

    def lastStepCambium(self, position1, position2):
        self.main.work("manual_cambium", (position1, position2))
        self.clickImage.registerClickEvent(lambda s,x: x)

    @debounce(0.5)
    def propagateValue(self, id, value):
        self.main.setValue(id, value)

    def makeButton(self, name, id):
        autocorrect = QPushButton(name)
        autocorrect.clicked.connect(partial(self.main.work, id))
        return autocorrect

    def makeSlider(self, name, id, factor, min, max):
        container = QWidget()
        container.setContentsMargins(0, 0, 0, 0)
        cLayout = QVBoxLayout()
        cLayout.setContentsMargins(0, 0, 0, 0)
        container.setLayout(cLayout)

        labelContainer = QWidget()
        labelContainer.setContentsMargins(0, 0, 0, 0)
        lLayout = QHBoxLayout()
        lLayout.setContentsMargins(0, 0, 0, 0)
        labelContainer.setLayout(lLayout)
        nameLabel = QLabel(name)
        lLayout.addWidget(nameLabel)
        vLabel = QLabel()
        vLabel.setAlignment(Qt.AlignRight)
        lLayout.addWidget(vLabel)
        alpha_slider = QSlider(Qt.Horizontal)
        alpha_slider.setMinimum(min * factor)
        alpha_slider.setMaximum(max * factor)
        alpha_slider.valueChanged.connect(partial(self.value_changed, id))
        self.widgets[id] = {
            "widget": alpha_slider,
            "type": "slider",
            "factor": factor,
            "label": vLabel,
        }
        cLayout.addWidget(labelContainer)
        cLayout.addWidget(alpha_slider)
        return container

    def setMain(self, main):
        self.main = main
        self.addTab(self.createCorrection(), "Image correction")
        self.addTab(self.createDetection(), "Cell detection")
        self.addTab(self.createStaining(), "Staining")
        self.addTab(self.createROI(), "ROI")
        self.addTab(self.createCounting(), "Counting")

    def onCurrentChange(self):
        ids = ["correction", "detection", "staining", "roi", "counting"]
        self.propagateValue("current", ids[self.currentIndex()])
        if ids[self.currentIndex()] == "counting":
            self.clickImage.registerClickEvent(partial(self.main.work, "count_click"))

    def makeAnalysis(self):
        s = self.main.settings["statistics"]
        factor = s["pixelsize"] * s["pixelsize"]
        if factor == 1:
            unit = " px²"
        else:
            unit = " µm²"
        results = {
            "Xylem cell count": s["countMatched"],
            "Secondary cell count": s["countMatched2"],
            "Total cell count": s["countMatched"] + s["countUnmatched"],
            "Xylem cell area": str(round(s["areaMatched"] * factor,2)) + unit,
            "Secondary cell area": str(round(s["areaMatched2"] * factor,2)) + unit,
            "Total cell area": str(round((s["areaMatched"] + s["areaUnmatched"]) * factor,2)) + unit,
            "Cambium area": str(round(s["circleArea"] * factor,2)) + unit,
        }
        self.analysis.setText(
            "<br>".join([("<em>" + k + ":</em> " + str(results[k])) for k in results])
        )

    def update(self):
        ids = {"correction": 0, "detection": 1, "staining": 2, "roi": 3, "counting": 4}
        for id in ids:
            self.setTabEnabled(ids[id], self.main.settings[id]["ready"])
        if self.isTabEnabled(ids[self.main.settings["current"]]):
            self.setCurrentIndex(ids[self.main.settings["current"]])
            if self.main.settings["current"] == "counting":
                self.clickImage.registerClickEvent(
                    partial(self.main.work, "count_click")
                )
        for widget in self.widgets:
            domain, id = widget.split("_")
            value = self.main.settings[domain][id]
            if self.widgets[widget]["type"] == "slider":
                self.widgets[widget]["label"].setText(
                    str(
                        round(
                            value,
                            math.ceil(math.log(self.widgets[widget]["factor"], 10)),
                        )
                    )
                )
                value *= self.widgets[widget]["factor"]
                value = round(value)
                sblock = self.widgets[widget]["widget"].blockSignals(True)
                if self.widgets[widget]["widget"].maximum() < value:
                    self.widgets[widget]["widget"].setMaximum(value)
                if self.widgets[widget]["widget"].minimum() > value:
                    self.widgets[widget]["widget"].setMinimum(value)
                self.widgets[widget]["widget"].setValue(value)
                self.widgets[widget]["widget"].blockSignals(sblock)
        self.makeAnalysis()
