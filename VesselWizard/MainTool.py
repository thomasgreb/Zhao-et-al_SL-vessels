import os, json, cv2, pickle, math, time
import numpy as np
from functools import partial
from PySide6.QtGui import QPixmap
import tifffile
from Worker import Worker
from debounce import debounce
import qimage2ndarray
from PySide6.QtCore import QRunnable, Slot, QThreadPool
import collections.abc


def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

#The main logic
class MainTool:
    image = None
    initial_settings = {
        "correction": {"ready": True, "alpha": 1, "beta": 0, "done": False},
        "detection": {
            "ready": False,
            "done": False,
            "binary": 200,
            "lower": 20,
            "higher": 99.5,
        },
        "staining": {
            "ready": False,
            "done": False,
            "lowerH": 80,
            "upperH": 110,
            "saturation": 10,
            "brightness": 160,
            "erosion": 1,
            "dilation": 2,
        },
        "roi": {
            "ready": False,
            "done": False,
            "erosion": 2,
            "blur": 50,
            "threshold": 64,
            "cutoff": 0.02,
        },
        "counting": {
            "ready": False,
            "done": False,
            "dilation": 0,
            "threshold": 0.75,
            "lower": 77,
            "higher": 99.9,
            "minCircularity": 30,
        },
        "statistics": {
            "countMatched": 0,
            "countMatched2": 0,
            "countUnmatched": 0,
            "areaMatched": 0,
            "areaMatched2": 0,
            "areaUnmatched": 0,
            "circleArea": 0,
            "pixelsize": 1,
        },
        "current": "correction",
        "size": {"x": 0, "y": 0, "aspect": 0},
    }

    current_view = {"zoom": 1, "x": 0, "y": 0}

    def __init__(self, clickImage, sidebar, setProgress, extraImage):
        self.sidebar = sidebar
        self.setProgress = setProgress
        self.clickImage = clickImage
        self.extraImage = extraImage
        self.sidebar.setMain(self)
        self.clickImage.setMain(self)
        self.threadpool = QThreadPool()

    def resetZoom(self):
        self.current_view = {"zoom": 1, "x": 0, "y": 0}

    def move(self, x, y, z):
        newX = self.current_view["x"]
        newY = self.current_view["y"]
        newZ = self.current_view["zoom"]
        currentWidth = self.settings["size"]["x"] * newZ
        currentHeight = self.settings["size"]["y"] * newZ
        if z != 0:
            newZ += z * 0.1
            newZ = max(min(newZ, 1), 0.1)
            newWidth = self.settings["size"]["x"] * newZ
            newHeight = self.settings["size"]["y"] * newZ
            newX += round((currentWidth - newWidth) / 2)
            newY += round((currentHeight - newHeight) / 2)
        else:
            newWidth = self.settings["size"]["x"] * newZ
            newHeight = self.settings["size"]["y"] * newZ
            if x != 0:
                newX += round(x * currentWidth / 6)
            if y != 0:
                newY += round(y * currentHeight / 6)
        newX = min(round(self.settings["size"]["x"] - newWidth), max(0, newX))
        newY = min(round(self.settings["size"]["y"] - newHeight), max(0, newY))
        self.current_view = {"zoom": newZ, "x": newX, "y": newY}
        self.clickImage.setCurrentView(self.current_view)
        self.extraImage.setCurrentView(self.current_view)
        self.setCurrentImage()

    def setImage(self, image):
        self.current_view = {"zoom": 1, "x": 0, "y": 0}
        self.image = image
        self.settings = self.initial_settings
        head, tail = os.path.split(self.image)
        self.folder = os.path.join(head, "metadata-" + tail)
        try:
            os.mkdir(self.folder)
        except FileExistsError as e:
            pass
        config = os.path.join(self.folder, "settings.json")
        if os.path.exists(config):
            with open(config, "r") as f:
                self.settings = update(self.settings, json.load(f))
        compressed = os.path.join(self.folder, "compressed.png")
        if not os.path.exists(compressed):
            imo = cv2.imread(self.image)
            aspect = imo.shape[1] / imo.shape[0]
            im = cv2.resize(
                imo, (2500, int(2500 / aspect)), interpolation=cv2.INTER_AREA
            )
            pixelsize = self.findPixelsize()
            self.setValue("size_xorg", imo.shape[1])
            self.setValue("size_yorg", imo.shape[0])
            self.setValue("size_x", im.shape[1])
            self.setValue("size_y", im.shape[0])
            self.setValue("size_aspect", im.shape[1] / im.shape[0])
            if pixelsize:
                self.setValue("statistics_pixelsize", pixelsize)
            cv2.imwrite(compressed, im)
        if os.path.exists(os.path.join(self.folder, "corrected.png")):
            self.corrected = cv2.imread(os.path.join(self.folder, "corrected.png"))
            if "size" not in self.settings or self.settings["size"]["aspect"] == 0:
                self.setValue("size_x", self.corrected.shape[1])
                self.setValue("size_y", self.corrected.shape[0])
                self.setValue(
                    "size_aspect", self.corrected.shape[1] / self.corrected.shape[0]
                )
        self.sidebar.update()
        self.setCurrentImage()

    def findPixelsize(self):
        try:
            with tifffile.TiffFile(self.image) as tif:
                for page in tif.pages:
                    for tag in page.tags:
                        tag_name, tag_value = tag.name, tag.value
                        if tag_name != "ImageDescription":
                            continue
                        for tagx in tag_value.split("|"):
                            try:
                                name, val = tagx.split(" = ")
                                if name != "MPP":
                                    continue
                                return float(val)
                            except:
                                pass
        except:
            return None

    def setCurrentImage(self, mat=None):
        if self.settings["correction"]["done"]:
            self.extraImage.setImage(os.path.join(self.folder, "corrected.png"))
        else:
            self.extraImage.setImage(os.path.join(self.folder, "compressed.png"))

        if mat is not None:
            self.clickImage.setMat(mat)
        elif self.settings["current"] == "correction":
            if self.settings["correction"]["done"]:
                self.clickImage.setImage(os.path.join(self.folder, "corrected.png"))
            else:
                self.clickImage.setImage(os.path.join(self.folder, "compressed.png"))
        elif self.settings["current"] == "detection":
            if self.settings["detection"]["done"]:
                self.clickImage.setImage(os.path.join(self.folder, "cells.png"))
            else:
                self.clickImage.setImage(os.path.join(self.folder, "corrected.png"))
        elif self.settings["current"] == "staining":
            if self.settings["staining"]["done"]:
                self.clickImage.setImage(os.path.join(self.folder, "staining.png"))
            else:
                self.clickImage.setImage(os.path.join(self.folder, "corrected.png"))
        elif self.settings["current"] == "roi":
            if self.settings["roi"]["done"]:
                self.clickImage.setImage(os.path.join(self.folder, "roi.png"))
            else:
                self.clickImage.setImage(os.path.join(self.folder, "corrected.png"))
        elif self.settings["current"] == "counting":
            res_file = os.path.join(self.folder, "result.png")
            if os.path.exists(res_file):
                self.clickImage.setImage(res_file)
            else:
                self.clickImage.setImage(os.path.join(self.folder, "corrected.png"))

    def work(self, command, data={}):
        self.data = data
        if command == "autocorrect":
            fn = self.autocorrect
        elif command == "image_correction":
            fn = self.image_correction
        elif command == "detect_cells":
            fn = self.detect_cells
        elif command == "analyze":
            fn = self.analyze
        elif command == "detect_staining":
            fn = self.detect_staining
        elif command == "detect_cambium":
            fn = self.detect_cambium
        elif command == "manual_cambium":
            fn = self.manual_cambium
        elif command == "count_click":
            fn = self.count_click
        elif command == "create_result":
            fn = self.create_result
        elif command == "detect_xylem":
            fn = self.detect_xylem
        worker = Worker(fn)  # Any other args, kwargs are passed to the run function
        worker.signals.result.connect(partial(self.result, command))
        worker.signals.progress.connect(partial(self.progress, command))
        worker.signals.error.connect(partial(self.error, command))
        self.threadpool.start(worker)

    def progress(self, command, progress):
        self.setProgress(progress)
        if command == "detect_xylem":
            self.work("create_result", {"progress": False})

    def result(self, command, result):
        self.data = {}
        self.setProgress(0)
        if command != "create_result" and len(result):
            for key in result:
                domain, id = key.split("_")
                self.settings[domain][id] = result[key]
            self.saveSettings()
            self.sidebar.update()
        if command == "autocorrect":
            self.work("image_correction")
        elif command == "image_correction":
            self.setCurrentImage()
        elif command == "detect_cells":
            self.setCurrentImage()
        elif command == "detect_staining":
            self.setCurrentImage()
        elif command == "detect_cambium":
            self.setCurrentImage()
        elif command == "manual_cambium":
            self.setCurrentImage()
        elif command == "count_click":
            self.work("create_result")
        elif command == "detect_xylem":
            self.work("analyze")
        elif command == "analyze":
            self.work("create_result", {"save": True})
        elif command == "create_result" and result is not None:
            self.setCurrentImage(result)
        # print(self.settings)

    def setValue(self, key, value):
        if not self.image:
            return
        try:
            domain, id = key.split("_")
            if len(id) > 0 and (domain not in self.settings):
                self.settings[domain] = {}
            self.settings[domain][id] = value
        except ValueError as e:
            self.settings[key] = value
        if key == "current":
            self.setCurrentImage()
        self.saveSettings()

    def error(self, command, error):
        self.data = {}
        self.setProgress(0)
        e, val, t = error
        self.sidebar.produceError(str(val))

    def saveSettings(self):
        with open(os.path.join(self.folder, "settings.json"), "w") as file:
            json.dump(self.settings, file)

    def autocorrect(self, progress):
        progress(-1)
        clip_hist_percent = 1
        image = cv2.imread(os.path.join(self.folder, "compressed.png"))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_size = len(hist)
        accumulator = []
        accumulator.append(float(hist[0]))
        for index in range(1, hist_size):
            accumulator.append(accumulator[index - 1] + float(hist[index]))
        maximum = accumulator[-1]
        clip_hist_percent *= maximum / 100.0
        clip_hist_percent /= 2.0
        minimum_gray = 0
        while accumulator[minimum_gray] < clip_hist_percent:
            minimum_gray += 1
        maximum_gray = hist_size - 1
        while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
            maximum_gray -= 1
        alpha = 255 / (maximum_gray - minimum_gray)
        beta = -minimum_gray * alpha
        return {"correction_alpha": alpha, "correction_beta": beta}

    def image_correction(self, progress):
        progress(-1)
        image = cv2.imread(os.path.join(self.folder, "compressed.png"))
        print("using", self.settings["correction"]["alpha"])
        auto_result = cv2.convertScaleAbs(
            image,
            alpha=self.settings["correction"]["alpha"],
            beta=self.settings["correction"]["beta"],
        )
        cv2.imwrite(os.path.join(self.folder, "corrected.png"), auto_result)
        self.corrected = auto_result
        return {"correction_done": True, "detection_ready": True}

    def detect_cells(self, progress):
        progress(-1)
        frame = self.corrected.copy()
        x, cell_mask = cv2.threshold(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
            int(self.settings["detection"]["binary"]),
            255,
            cv2.THRESH_BINARY,
        )
        numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            cell_mask, 4, cv2.CV_32S
        )
        areas = [x for x in stats[:, cv2.CC_STAT_AREA] if x > 5]
        lower = np.percentile(areas, self.settings["detection"]["lower"])
        higher = np.percentile(areas, self.settings["detection"]["higher"])
        area_map = stats[:, cv2.CC_STAT_AREA]
        size_map = area_map[labels]
        size_map[(size_map < lower) | (size_map > higher)] = 0
        size_map[size_map > 0] = 1
        labels *= size_map
        with open(os.path.join(self.folder, "areas.pickle"), "wb") as file:
            pickle.dump(area_map, file)
        with open(os.path.join(self.folder, "cells.pickle"), "wb") as file:
            pickle.dump(labels, file)
        with open(os.path.join(self.folder, "xylem.pickle"), "wb") as file:
            pickle.dump(set(), file)
        with open(os.path.join(self.folder, "sizemap.pickle"), "wb") as file:
            pickle.dump(size_map, file)
        with open(os.path.join(self.folder, "centroids.pickle"), "wb") as file:
            pickle.dump(centroids, file)
        colors_decimal = ((np.mod(labels, 6) + 1) * size_map).astype(np.uint8)
        colors_binary = (
            np.unpackbits(colors_decimal, axis=None).reshape(
                (colors_decimal.shape[0], colors_decimal.shape[1], 8), order="C"
            )[:, :, -3:]
        ) * 255
        bwframe = cv2.cvtColor(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR
        )
        merged = cv2.addWeighted(bwframe, 0.5, colors_binary, 0.5, 0)
        cv2.imwrite(os.path.join(self.folder, "cells.png"), merged)
        return {"detection_done": True, "staining_ready": True}

    def detect_staining(self, progress):
        progress(-1)
        frame = self.corrected.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_blue = np.array(
            [
                int(self.settings["staining"]["lowerH"]),
                int(self.settings["staining"]["saturation"]),
                int(self.settings["staining"]["brightness"]),
            ]
        )
        upper_blue = np.array([int(self.settings["staining"]["upperH"]), 255, 255])
        xylem_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        if self.settings["staining"]["erosion"] > 0:
            erosion_size = int(self.settings["staining"]["erosion"]) - 1
            erosion_element = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * erosion_size + 1, 2 * erosion_size + 1),
                (erosion_size, erosion_size),
            )
            xylem_mask = cv2.erode(xylem_mask, erosion_element)
        if self.settings["staining"]["dilation"] > 0:
            dilation_size = int(self.settings["staining"]["dilation"]) - 1
            dilation_element = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * dilation_size + 1, 2 * dilation_size + 1),
                (dilation_size, dilation_size),
            )
            xylem_mask = cv2.dilate(xylem_mask, dilation_element)
        with open(os.path.join(self.folder, "sizemap.pickle"), "rb") as file:
            size_map = pickle.load(file).astype(np.uint8)
        cv2.imwrite(os.path.join(self.folder, "xylem_mask.png"), xylem_mask)
        cell_walls = cv2.bitwise_not(size_map)
        xylem_mask *= cell_walls
        cells_white = cv2.cvtColor(size_map * 255, cv2.COLOR_GRAY2BGR)
        xylem_white = cv2.cvtColor(xylem_mask, cv2.COLOR_GRAY2BGR)
        # print(np.max(xylem_mask), np.max(xylem_white))
        frame[cells_white == 255] = 255
        frame[xylem_white == 1] = 0
        xylem_white *= 255
        xylem_white[:, :, 0:2] = 0
        frame = cv2.add(frame, xylem_white)
        cv2.imwrite(os.path.join(self.folder, "staining.png"), frame)
        return {"staining_done": True, "roi_ready": True}

    def detect_cambium(self, progress):
        progress(-1)
        xylem_mask = cv2.cvtColor(
            cv2.imread(os.path.join(self.folder, "xylem_mask.png")), cv2.COLOR_BGR2GRAY
        )
        if self.settings["roi"]["erosion"] > 0:
            dilation_size = int(self.settings["roi"]["erosion"]) - 1
            dilation_element = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * dilation_size + 1, 2 * dilation_size + 1),
                (dilation_size, dilation_size),
            )
            xylem_mask = cv2.erode(xylem_mask, dilation_element)
        x, cambium_search = cv2.threshold(
            cv2.blur(
                xylem_mask,
                (int(self.settings["roi"]["blur"]), int(self.settings["roi"]["blur"])),
            ),
            int(self.settings["roi"]["threshold"]),
            255,
            cv2.THRESH_BINARY,
        )
        M = cv2.moments(cambium_search)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        y, x = np.indices((cambium_search.shape))
        r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
        r = r.astype(int)
        tbin = np.bincount(r.ravel(), cambium_search.ravel())
        nr = np.bincount(r.ravel())
        radialprofile = tbin / nr
        kernel_size = 100
        kernel = np.ones(kernel_size) / kernel_size
        data_convolved_10 = np.convolve(radialprofile, kernel, mode="same")
        data_convolved_10 /= np.max(data_convolved_10)
        data_convolved_10 *= 1.05
        data_convolved_10 -= 0.05
        zero_crossing = np.where(np.diff(np.signbit(data_convolved_10)))[0]
        if (
            zero_crossing.size > 0
            and zero_crossing[0] / xylem_mask.shape[0] > self.settings["roi"]["cutoff"]
        ):
            radius = int(zero_crossing[0] * 1.1)
            with open(os.path.join(self.folder, "cambium.pickle"), "wb") as file:
                pickle.dump((center, radius), file)
            image = self.corrected.copy()
            cv2.circle(image, (int(center[0]), int(center[1])), radius, (0, 0, 255), 10)
            cv2.imwrite(os.path.join(self.folder, "roi.png"), image)
            return {
                "roi_done": True,
                "counting_ready": True,
                "statistics_circleArea": np.pi * radius * radius,
            }
        else:
            raise Exception("Could not autodetect cambium. Please do it manually.")

    def manual_cambium(self, progress):
        progress(-1)
        centerP, cambiumP = self.data
        image = self.corrected.copy()
        h, w, c = image.shape
        # print(centerP, w, h)
        center = (w * centerP[0], h * centerP[1])
        cambium = (w * cambiumP[0], h * cambiumP[1])
        # print(center, cambium)
        radius = math.dist(center, cambium)
        with open(os.path.join(self.folder, "cambium.pickle"), "wb") as file:
            pickle.dump(((int(center[0]), int(center[1])), int(radius)), file)
        image = cv2.circle(
            image, (int(center[0]), int(center[1])), int(radius), (0, 0, 255), 10
        )
        cv2.imwrite(os.path.join(self.folder, "roi.png"), image)
        return {
            "roi_done": True,
            "counting_ready": True,
            "statistics_circleArea": np.pi * radius * radius,
        }

    def count_click(self, progress):
        progress(-1)
        x, y, secondary = self.data
        with open(os.path.join(self.folder, "cells.pickle"), "rb") as file:
            labels = pickle.load(file)
        if secondary:
            try:
                with open(os.path.join(self.folder, "xylem2.pickle"), "rb") as file:
                    xylem = pickle.load(file)
            except Exception as e:
                xylem = set()
        else:
            with open(os.path.join(self.folder, "xylem.pickle"), "rb") as file:
                xylem = pickle.load(file)
        h, w = labels.shape
        x = int(x * w)
        y = int(y * h)
        id = labels[y, x]
        if id == 0:
            return {}
        if id in xylem:
            # print(id)
            xylem.remove(id)
        else:
            xylem.add(id)
        if secondary:
            with open(os.path.join(self.folder, "xylem2.pickle"), "wb") as file:
                pickle.dump(xylem, file)
        else:
            with open(os.path.join(self.folder, "xylem.pickle"), "wb") as file:
                pickle.dump(xylem, file)
        return {}

    def analyze(self, progress):
        statistics = {
            "statistics_countMatched": 0,
            "statistics_countMatched2": 0,
            "statistics_countUnmatched": 0,
            "statistics_areaMatched": 0,
            "statistics_areaMatched2": 0,
            "statistics_areaUnmatched": 0,
        }
        progress(-1)
        with open(os.path.join(self.folder, "xylem.pickle"), "rb") as file:
            xylem_set = pickle.load(file)
        try:
            with open(os.path.join(self.folder, "xylem2.pickle"), "rb") as file:
                xylem2_set = pickle.load(file)
        except Exception as e:
            xylem2_set = set()
        with open(os.path.join(self.folder, "cells.pickle"), "rb") as file:
            labels = pickle.load(file)
        with open(os.path.join(self.folder, "areas.pickle"), "rb") as file:
            areas = pickle.load(file)
        label_list = np.unique(labels)
        label_count = len(label_list) - 1
        x = 0
        for i in label_list:
            area = areas[i]
            if i in xylem2_set:
                statistics["statistics_countMatched2"] += 1
                statistics["statistics_areaMatched2"] += float(area)
            if i not in xylem_set:
                statistics["statistics_countUnmatched"] += 1
                statistics["statistics_areaUnmatched"] += float(area)
                continue
            statistics["statistics_countMatched"] += 1
            statistics["statistics_areaMatched"] += float(area)
            progress(int((x / label_count) * 100))
        return statistics

    def create_result(self, progress):
        if "save" in self.data:
            save = self.data["save"]
        else:
            save = False
        if "progress" in self.data:
            showProgress = self.data["progress"]
        else:
            showProgress = True
        if showProgress:
            progress(-1)
        d = time.time_ns()
        with open(os.path.join(self.folder, "cells.pickle"), "rb") as file:
            labels = pickle.load(file)
        try:
            with open(os.path.join(self.folder, "xylem.pickle"), "rb") as file:
                xylem_set = pickle.load(file)
        except Exception as e:
            return None
        try:
            with open(os.path.join(self.folder, "xylem2.pickle"), "rb") as file:
                xylem_set2 = pickle.load(file)
        except Exception as e:
            xylem_set2 = set()
        if len(xylem_set) == 0 and len(xylem_set2) == 0:
            return cv2.cvtColor(self.corrected, cv2.COLOR_BGR2RGB)
        # print("loaded",time.time_ns())

        colors = [
            [xylem_set.difference(xylem_set2), [0, 0, 255]],
            [xylem_set2.difference(xylem_set), [0, 255, 0]],
            [xylem_set.intersection(xylem_set2), [0, 255, 255]],
        ]

        image = self.corrected.copy()

        for xset, col in colors:
            if len(xset) == 0:
                continue
            xylem = np.sort(np.fromiter(xset, int, len(xset)))
            #print("dfff",xylem)
            idx = np.searchsorted(xylem, labels.ravel())
            idx[idx == len(xylem)] = 0
            xlabels = labels.copy()
            xlabels[xylem[idx].reshape(xlabels.shape) != xlabels] = 0
            image[xlabels > 0] = col

        if save:
            cv2.imwrite(os.path.join(self.folder, "result.png"), image)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def detect_xylem(self, progress):
        progress(-1)
        with open(os.path.join(self.folder, "cells.pickle"), "rb") as file:
            labels = pickle.load(file)
        with open(os.path.join(self.folder, "centroids.pickle"), "rb") as file:
            centroids = pickle.load(file)
        with open(os.path.join(self.folder, "cambium.pickle"), "rb") as file:
            center, radius = pickle.load(file)
        with open(os.path.join(self.folder, "areas.pickle"), "rb") as file:
            areas = pickle.load(file)
        areass = [x for x in areas if x > 5]
        lower = np.percentile(areass, self.settings["counting"]["lower"])
        higher = np.percentile(areass, self.settings["counting"]["higher"])
        xylem_mask = cv2.imread(
            os.path.join(self.folder, "xylem_mask.png"), cv2.IMREAD_GRAYSCALE
        )
        label_list = np.unique(labels)
        label_count = len(label_list) - 1
        if self.settings["counting"]["dilation"] > 0:
            dilate = True
            dilation_size = self.settings["counting"]["dilation"] - 1
            dilation_element = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * dilation_size + 1, 2 * dilation_size + 1),
                (dilation_size, dilation_size),
            )
        else:
            dilate = False
        xylem = set()
        x = 0
        for i in label_list:
            if i == 0:
                continue
            x += 1
            area = areas[i]
            if area < lower or area > higher:
                continue
            if np.linalg.norm(center - centroids[i]) > radius:
                continue
            component = (labels == i).astype(np.uint8) * 255
            if dilate:
                component_dilated = cv2.dilate(component, dilation_element)
            contours, h = cv2.findContours(
                component_dilated if dilate else component,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_NONE,
            )
            perimeter = cv2.arcLength(contours[0], True)
            circularity = (area / ((perimeter * perimeter) / (4 * np.pi))) * 100
            if self.settings["counting"]["minCircularity"] > circularity:
                continue
            sum = 0
            count = 0
            for j in range(0, len(contours[0]), 2):
                count += 1
                sum += (xylem_mask[contours[0][j][0][1], contours[0][j][0][0]]) / 255
            if count < 1:
                continue
            xylem_ratio = sum / count
            if xylem_ratio < self.settings["counting"]["threshold"]:
                continue
            xylem.add(i)
            with open(os.path.join(self.folder, "xylem.pickle"), "wb") as file:
                pickle.dump(xylem, file)
            progress(int((x / label_count) * 100))
        return {"counting_done": True}
