#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

import numpy as np

class BoDataSet:
    def __init__(self, trainPath=None, validPath = None, testPath = None):
        if not trainPath:
            trainPath = "./boQuatrain5/ptrain"
        if not validPath:
            validPath = "./boQuatrain5/pvalid"
        if not testPath:
            testPath = "./boQuatrain5/ptest"
        self.trainPath = trainPath
        self.validPath = validPath
        self.testPath = testPath
    def getTrainData():
