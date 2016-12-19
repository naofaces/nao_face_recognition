import sys
from PyQt4.QtCore import Qt
from PyQt4.QtGui import *
from PyQt4.QtCore import QAbstractFileEngine
from PyQt4.QtCore import QAbstractFileEngineHandler
from PyQt4.QtCore import QFSFileEngine

class FileDialogHandler(QAbstractFileEngineHandler):
    def create(self,filename):
        if str(filename).startswith(':'):
            return None # Will be handled by Qt as a resource file
        print("Create QFSFileEngine for {0}".format(filename))
        return QFSFileEngine(filename)

class Example(QMainWindow):

    def __init__(self):
        super(Example, self).__init__()

        self.initUI()

    def initUI(self):

        self.textEdit = QTextEdit()
        self.setCentralWidget(self.textEdit)
        self.statusBar()

        openFile = QAction(QIcon('open.png'), 'Open', self)
        openFile.setShortcut('Ctrl+O')
        openFile.setStatusTip('Open new File')
        openFile.triggered.connect(self.showDialog)

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(openFile)

        self.setGeometry(300, 300, 350, 300)
        self.setWindowTitle('File dialog')
        self.show()

    def showDialog(self):
        handler = FileDialogHandler()
        #using QFileDialog.getOpenFileName works fine
        fname = QFileDialog.getOpenFileName(None, 'Open file', '/home','All files (*.*)', options=QFileDialog.DontUseNativeDialog)
        #dialog = QFileDialog()
        #dialog.setOption(QFileDialog.DontUseNativeDialog,False)
        #if dialog.exec_():
            #fname = dialog.selectedFiles()
        #else:
            #fname = None
        f = open(fname, 'r')
        with f:
            data = f.read()
            self.textEdit.setText(data)

def main():
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()