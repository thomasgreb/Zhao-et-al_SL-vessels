from PySide6.QtCore import QRunnable, Slot, Signal, QObject

import sys
import traceback


class WorkerSignals(QObject):
    error = Signal(tuple)
    result = Signal(object)
    progress = Signal(int)




class Worker(QRunnable):
    def __init__(self, fn):
        super(Worker, self).__init__()
        self.fn = fn
        self.signals = WorkerSignals()

    @Slot()
    def run(self):
        try:
            result = self.fn(progress=self.signals.progress.emit)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)