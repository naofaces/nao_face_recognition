import sys
import time

from PyQt4.QtCore import (QCoreApplication, QObject, QRunnable, QThread,
                          QThreadPool, pyqtSignal)


# Subclassing QObject and using moveToThread
# http://blog.qt.digia.com/blog/2007/07/05/qthreads-no-longer-abstract
class SomeObject(QObject):

    finished = pyqtSignal()

    def long_running(self):
        count = 0
        while count < 5:
            time.sleep(1)
            print("B Increasing")
            count += 1
        self.finished.emit()


def using_move_to_thread():
    app = QCoreApplication([])
    objThread = QThread()
    obj = SomeObject()
    obj.moveToThread(objThread)
    obj.finished.connect(objThread.quit)
    objThread.started.connect(obj.long_running)
    objThread.finished.connect(app.exit)
    objThread.start()
    sys.exit(app.exec_())


if __name__ == "__main__":
    using_move_to_thread()