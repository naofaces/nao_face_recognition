#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This package implements face recognition for NAO robots. The Openface framework is used for recognition tasks, working
on images retrieved by Nao. The framework can be found here:
https://cmusatyalab.github.io/openface/
The user interface, recognition as well as training processes is a python adaptation of the Openface Real-time Web
Demo: https://cmusatyalab.github.io/openface/demo-1-web/

This script is used to start the application.
"""

import sys
from PyQt4 import QtGui, QtCore
from PyQt4.QtGui import QImage, QLabel, QPixmap, QFileDialog
from PyQt4.QtCore import QThread
from functools import partial
import face_recognition_config as config
import face_recognition_methods
import greet_persons
# suppress sklearn deprecation warnings
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

naoqi_root = config.naoqi_root
sys.path.insert(0, config.naoqi_root)
# naoqi will be found at runtime if installed in config.naoqi_root
from naoqi import ALProxy


class FaceRecognitionGui(QtGui.QMainWindow):
    """
    This class implements the user interface and starts the worker thread responsible for image retrieval and face
    recognition.
    """
    # class for generating the user interface
    robotIp = config.ip
    port = 9559

    greeter = None
    greeting_mode = False

    # holds nao images for display
    pix_holder = []
    # text area for person names
    text_area = None

    '''
    Openface variables
    Shared between GUI and thread, therefore lists have to be used (working as references)
    '''
    training = [False]
    people = []
    image_count_persons = []
    # current nao image
    current_image = []
    current_result_name = []

    '''
    Variables for thread components
    Prevents garbage collection of thread
    '''
    face_recognition_thread = None
    worker = None

    def __init__(self, win_parent=None):
        QtGui.QMainWindow.__init__(self, win_parent)
        # start face recognition pipeline in separate thread
        self.create_thread()
        self.create_widgets()
        self.setWindowTitle('NAO Face Recognition')
        self.current_result_name.append(None)
        # Initializing object for greeting mode
        self.greeter = greet_persons.Greeter()
        # timer event occurs once every time when there a no other window events
        self.startTimer(0)

    def create_thread(self):
        """
        Creates the thread used for the recognition pipeline
        """
        obj_thread = QThread()
        obj = face_recognition_methods.FaceRecognitionWorker(self.current_image, self.current_result_name,
                                                             self.image_count_persons, self.people)
        obj.moveToThread(obj_thread)
        # TODO: Unsubscribe from videoproxy here?
        # Sample code
        # main_window.videoProxy.unsubscribe(main_window.imgClient)
        # obj.finished.connect(obj_thread.quit)
        obj_thread.started.connect(obj.image_processing)
        obj_thread.start()
        self.face_recognition_thread = obj_thread
        self.worker = obj

    '''
    Creating the main window elements
    '''

    def create_widgets(self):
        # all components are added to main_grid
        main_grid = QtGui.QGridLayout()
        main_grid.addWidget(QLabel("<b>Camera Stream:</b>"))

        # the current nao image
        self.current_image.append(QImage())
        self.create_menu()
        # panels for image display, controls, and output about people and no. of images
        self.create_stream_panel(main_grid)
        self.create_control_panel(main_grid)
        self.create_text_panel(main_grid)

        # Create central widget
        central_widget = QtGui.QWidget()
        central_widget.setLayout(main_grid)
        self.setCentralWidget(central_widget)

    def create_menu(self):
        # creates the file menu
        open_action = QtGui.QAction('&Open', self)
        open_action.setShortcut('Ctrl+O')
        open_action.triggered.connect(self.open_file)

        save_action = QtGui.QAction('&Save', self)
        save_action.setShortcut('Ctrl+S')
        save_action.triggered.connect(self.save_file)

        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu('&File')
        file_menu.addAction(open_action)
        file_menu.addAction(save_action)

    def save_file(self):
        # It is recommended to save pickle files with a *.p extension
        file_name = QFileDialog.getSaveFileName(None, 'Save file', '/home', 'Pickle files (*.p)',
                                                options=QFileDialog.DontUseNativeDialog)

        if file_name:
            print 'INFO: Saving ' + str(file_name)
            self.worker.save_model(file_name)

    def open_file(self):
        file_name = QFileDialog.getOpenFileName(None, 'Open file', '/home', 'Pickle files (*.p)',
                                                options=QFileDialog.DontUseNativeDialog)
        if file_name:
            print 'INFO: Opening ' + str(file_name)
            self.worker.open_model(file_name)

    def create_stream_panel(self, main_grid):
        # display for nao images
        stream_grid = QtGui.QGridLayout()
        stream_label = QLabel("Stream")
        stream_grid.addWidget(stream_label)
        self.pix_holder.append(stream_label)
        c = QtGui.QWidget()
        c.setLayout(stream_grid)
        main_grid.addWidget(c, 1, 0)

    def create_control_panel(self, main_grid):
        # holds controls for adding persons, toggling training mode and greeting mode
        control_grid = QtGui.QGridLayout()

        name_input_field = QtGui.QLineEdit()
        add_button = QtGui.QPushButton("Add Person")
        training_button = QtGui.QPushButton("Turn training on")
        greet_button = QtGui.QPushButton("Turn greetings on")
        # Send a signal to worker thread that training was switched on/off
        QtCore.QObject.connect(training_button, QtCore.SIGNAL("clicked()"), partial(self.worker.training_response,
                                                                                    training_param=self.training))
        button_list = list()
        button_list.append(add_button)
        button_list.append(training_button)
        button_list.append(greet_button)

        control_grid.addWidget(name_input_field, 0, 0)
        control_grid.addWidget(add_button, 0, 1)
        control_grid.addWidget(training_button, 1, 1)
        control_grid.addWidget(greet_button, 2, 1)

        # Connecting buttons with handler methods
        QtCore.QObject.connect(add_button, QtCore.SIGNAL("pressed()"),
                               partial(self.handler_add_person, input_field=name_input_field))
        QtCore.QObject.connect(training_button, QtCore.SIGNAL("pressed()"),
                               partial(self.handler_training, button=training_button))
        QtCore.QObject.connect(greet_button, QtCore.SIGNAL("pressed()"),
                               partial(self.handler_greet, button=greet_button))

        c = QtGui.QWidget()
        c.setLayout(control_grid)
        main_grid.addWidget(c, 2, 0)

    def create_text_panel(self, main_grid):
        # displays information about trained people and no. of images
        list_grid = QtGui.QGridLayout()
        self.text_area = QtGui.QTextEdit()
        self.text_area.setReadOnly(True)
        list_grid.addWidget(QLabel("<b>Persons:</b>"))

        list_grid.addWidget(self.text_area)
        c = QtGui.QWidget()
        c.setLayout(list_grid)
        main_grid.addWidget(c, 3, 0)

    '''
    Update methods for gui
    '''

    def update_text_area(self):
        # updates information about trained people and no. of images
        text = ''
        for name in self.people:
            index = self.people.index(name)
            if index > (len(self.image_count_persons) - 1) or len(self.image_count_persons) == 0:
                count = 0
            else:
                count = self.image_count_persons[index]
            text += name + ' (Images: ' + str(count) + ')\n'
        self.text_area.setText(text)
        self.text_area.verticalScrollBar().setValue(self.text_area.verticalScrollBar().maximum())

    def timerEvent(self, event):
        # Periodically check for greetings
        if self.greeting_mode:
            self.greeter.check_greeting(self.current_result_name[0])
        # Called periodically. Updates entire gui
        self.update()

    def paintEvent(self, event):
        # The image to be displayed is created by the worker thread and place in current_image[0]
        if self.pix_holder[0]:
            # setPixmap draws the image on screen
            self.pix_holder[0].setPixmap(QPixmap.fromImage(self.current_image[0]))
        self.update_text_area()

    '''
    Event handler for gui
    '''

    def handler_add_person(self, input_field):
        # adds a name to the list of trained persons
        if unicode(input_field.text().toUtf8(), encoding="UTF-8").lstrip():
            name = unicode(input_field.text().toUtf8(), encoding="UTF-8").lstrip()
            print 'INFO: Add Person: ' + name
            self.people.append(name)
            input_field.setText('')

    def handler_training(self, button):
        # Toggles the training variable which is held by the gui, and the button label
        if not self.training[0]:
            self.training[0] = True
            button.setText('Turn training off')
        else:
            self.training[0] = False
            button.setText('Turn training on')
        print 'INFO: Training set to ' + str(self.training[0])

    def handler_greet(self, button):
        # Toggles greeting mode on and off
        if not self.greeting_mode:
            self.greeting_mode = True
            button.setText('Turn greetings off')
        else:
            self.greeting_mode = False
            button.setText('Turn greetings on')
        print 'INFO: Greeting mode set to ' + str(self.greeting_mode)


if __name__ == '__main__':
    try:
        app = QtGui.QApplication(sys.argv)
        main_window = FaceRecognitionGui()
        main_window.show()
        app.exec_()
        print 'INFO: End program'
    except KeyboardInterrupt:
        print 'INFO: Interrupted by user, shutting down'
        sys.exit(0)
