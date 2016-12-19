#!/usr/bin/python
# -*- coding: utf-8 -*-

from PyQt4 import QtCore
from PyQt4.QtGui import QImage
from PyQt4.QtCore import pyqtSignal, QObject, pyqtSlot
import sys
import Image
import ImageDraw
import time
import face_recognition_config as config

# Openface imports
import argparse
import cv2
import imagehash
import json
import numpy as np
import os
import StringIO
import urllib
import base64

from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.manifold import TSNE
from sklearn.svm import SVC

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

naoqi_root = config.naoqi_root
openface_root = config.openface_root
sys.path.insert(0, config.naoqi_root)
sys.path.insert(0, config.openface_root)
# naoqi will be found at runtime if installed in config.naoqi_root
from naoqi import ALProxy

modelDir = os.path.join(config.openface_root, 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')
import openface

parser = argparse.ArgumentParser()
parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                    default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
parser.add_argument('--networkModel', type=str, help="Path to Torch network model.",
                    default=os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'))
parser.add_argument('--imgDim', type=int,
                    help="Default image dimension.", default=96)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--unknown', type=bool, default=False,
                    help='Try to predict unknown people')

args = parser.parse_args()

align = openface.AlignDlib(args.dlibFacePredictor)
net = openface.TorchNeuralNet(args.networkModel, imgDim=args.imgDim,
                              cuda=args.cuda)


class FaceRecognitionWorker(QtCore.QObject):
    """
    This class handles image retrieval and the entire training and recognition pipeline. It creates annotated images to
    be displayed in the gui.
    """
    notification = pyqtSignal(QObject)
    robotIp = config.ip
    port = 9559
    imgClient = None
    videoProxy = None
    # current nao image
    current_image = []
    current_result_name = []

    def __init__(self, image_for_gui, name_for_gui, image_count_for_gui, parent=None):
        print 'Initializing face recognition worker'
        super(FaceRecognitionWorker, self).__init__(parent)
        self.connect_to_nao()
        self.current_image = image_for_gui
        self.current_result_name = name_for_gui
        # number of training images per person
        self.image_count_persons = image_count_for_gui

        # Openface variables
        self.images = {}
        self.training = False
        self.people = []

        # Check if new person was added
        self.len_people_old = 0

        self.svm = None
        # if args.unknown:
        #     self.unknownImgs = np.load("./examples/web/unknown.npy")

    def connect_to_nao(self):
        try:
            # tts = ALProxy('ALTextToSpeech', self.robotIp, self.port)
            self.videoProxy = ALProxy("ALVideoDevice", self.robotIp, self.port)
            resolution = 2  # VGA
            color_space = 11  # RGB
            frame_rate = 30
            self.imgClient = self.videoProxy.subscribe("imgclient", resolution, color_space, frame_rate)
        except Exception, e:
            'Could not create proxy: ' + str(e)

    @pyqtSlot()
    def image_processing(self):
        print 'Starting image processing'
        while True:
            # Retrieve a new image from Nao.
            nao_image = self.videoProxy.getImageRemote(self.imgClient)
            # TODO: proper error handling
            if nao_image is not None:
                # getImageRemote returns images as lists of data
                width = nao_image[0]
                height = nao_image[1]
                image_data = nao_image[6]
                # Create a PIL Image from our pixel array.
                rgb_image = Image.frombytes("RGB", (width, height), image_data)
            else:
                rgb_image = Image.open('failure.jpeg')
                draw = ImageDraw.Draw(rgb_image)
                draw.text((5, 10), 'ERROR: Could not get image from nao, pls reboot nao. Okithxbye')
                width, height = rgb_image.size
                print 'ERROR: Could not get image from nao, pls reboot nao. Okithxbye'

            # print type(rgb_image)
            # if self.training:
            #     f = Face(np.array(rgb_image), self.people[-1])
            #     self.images.append(f)

            # Process the frame using Openface. Each frame is either used for training, or for recognition
            rgb_image, returned_name = self.processFrame(rgb_image, len(self.people)-1)
            if self.training:
                # Calculate the number of images per person
                current_image_count = len(self.images)  # Get total number of training images
                # Sum of training images per person in total, except for current one
                image_sum = sum(self.image_count_persons[:-1])
                # Number of training images for current person
                current_image_count = current_image_count - image_sum
                self.image_count_persons[-1] = current_image_count

            # draw = ImageDraw.Draw(rgb_image)
            # # font = ImageFont.load('arial.pil')
            # # draw.line((0, 0) + rgb_image.size, fill = (128, 128, 0, 0))
            # # draw.rectangle((0, 0, 100, 100), fill = (0, 170, 0, 0))
            # if self.training:
            #     draw.text((5, 10), 'Training ON')
            # else:
            #     draw.text((5, 10), 'Training OFF')
            # del draw

            # Transforming the image to QImage for gui
            self.current_image[0] = QImage(rgb_image.tobytes(), width, height, QImage.Format_RGB888)
            self.current_result_name[0] = returned_name

            time.sleep(0.01)

    @pyqtSlot()
    def training_response(self, training_param, people_param):
        # Reacts to toggling the training in GUI
        print 'Training set to ' + str(training_param[0])
        self.training = training_param[0]
        self.people = people_param

        if self.training and self.len_people_old != len(self.people):
            self.len_people_old += 1
            self.image_count_persons.append(0)
        print 'Persons:'
        for name in people_param:
            print name
        if not self.training:
            self.trainSVM()

    """
    Openface code, based on https://cmusatyalab.github.io/openface/demo-1-web/
    """
    def getData(self):
        X = []
        y = []
        for img in self.images.values():
            X.append(img.rep)
            y.append(img.identity)
        # identity -1 = unknown????? D:
        numIdentities = len(set(y + [-1])) - 1
        if numIdentities == 0:
            return None

        if args.unknown:
            numUnknown = y.count(-1)
            numIdentified = len(y) - numUnknown
            numUnknownAdd = (numIdentified / numIdentities) - numUnknown
            if numUnknownAdd > 0:
                print("+ Augmenting with {} unknown images.".format(numUnknownAdd))
                for rep in self.unknownImgs[:numUnknownAdd]:
                    # print(rep)
                    X.append(rep)
                    y.append(-1)

        X = np.vstack(X)
        y = np.array(y)
        return (X, y)

    def trainSVM(self):
        print("+ Training SVM on {} labeled images.".format(len(self.images)))
        d = self.getData()
        if d is None:
            self.svm = None
            return
        else:
            (X, y) = d
            numIdentities = len(set(y + [-1]))
            if numIdentities <= 1:
                return

            param_grid = [
                {'C': [1, 10, 100, 1000],
                 'kernel': ['linear']},
                {'C': [1, 10, 100, 1000],
                 'gamma': [0.001, 0.0001],
                 'kernel': ['rbf']}
            ]
            self.svm = GridSearchCV(SVC(C=1), param_grid, cv=5).fit(X, y)

    def processFrame(self, img, identity):
        is_training = self.training
        result_name = None
        #img = img.resize((300, 400), Image.ANTIALIAS)
        buf = np.fliplr(np.asarray(img))
        width, height = img.size
        #rgbFrame = np.zeros((height, width, 3), dtype=np.uint8)
        rgbFrame = np.zeros((height, width, 3), dtype=np.uint8)
        rgbFrame[:, :, 0] = buf[:, :, 2]
        rgbFrame[:, :, 1] = buf[:, :, 1]
        rgbFrame[:, :, 2] = buf[:, :, 0]
        if not is_training:
            annotatedFrame = np.copy(buf)
        identities = []
        bb = align.getLargestFaceBoundingBox(rgbFrame)
        bbs = [bb] if bb is not None else []
        for bb in bbs:
            landmarks = align.findLandmarks(rgbFrame, bb)
            alignedFace = align.align(args.imgDim, rgbFrame, bb,
                                      landmarks=landmarks,
                                      landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
            if alignedFace is None:
                # Return to beginning of loop
                continue

            phash = str(imagehash.phash(Image.fromarray(alignedFace)))
            if phash in self.images:
                identity = self.images[phash].identity
            else:
                rep = net.forward(alignedFace)
                if is_training:
                    self.images[phash] = Face(rep, identity)
                    # TODO msg schicken?
                    # content = [str(x) for x in alignedFace.flatten()]
                else:
                    if len(self.people) == 0:
                        identity = -1
                    elif len(self.people) == 1:
                        identity = 0
                    elif self.svm:
                        identity = self.svm.predict(rep)[0]
                    else:
                        identity = -1
                    if identity not in identities:
                        identities.append(identity)

            if not is_training:
                bl = (bb.left(), bb.bottom())
                tr = (bb.right(), bb.top())
                cv2.rectangle(annotatedFrame, bl, tr, color=(153, 255, 204),
                              thickness=3)
                for p in openface.AlignDlib.OUTER_EYES_AND_NOSE:
                    cv2.circle(annotatedFrame, center=landmarks[p], radius=3,
                               color=(102, 204, 255), thickness=-1)
                if identity == -1:
                    if len(self.people) == 1:
                        name = self.people[0]

                    else:
                        name = "Unknown"
                else:
                    name = self.people[identity]
                result_name = name
                cv2.putText(annotatedFrame, name, (bb.left(), bb.top() - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75,
                            color=(152, 255, 204), thickness=2)
        if not is_training:
            processed_image = Image.fromarray(annotatedFrame, 'RGB')
        else:
            processed_image = Image.fromarray(buf, 'RGB')
        return processed_image, result_name


class Face:
    def __init__(self, rep, identity):
        self.rep = rep
        self.identity = identity

    def __repr__(self):
        return "{{id: {}, rep[0:5]: {}}}".format(
            str(self.identity),
            self.rep[0:5])
