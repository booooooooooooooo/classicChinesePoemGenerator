import tensorflow as tf
import sys
import os
from sets import Set
import numpy as np

from data_util import prepare_Generator_RNN_Data
from general_util import get_minibatches
from scipy import spatial


class Config_Generator_RNN(object):
    def __init__(self, state_size, lr, n_epochs,batch_size, dirToSaveModel, dirToLog):
        self.state_size = state_size
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size  = batch_size
        self.dirToSaveModel = dirToSaveModel
        self.dirToLog = dirToLog
    def getStringOfParas(self):
        return "state_size_{:}_lr_{:}_n_epochs_{:}_batch_size_{:}".formate(self.state_size, self.lr, self.n_epochs, self.batch_size)

class Generator_RNN(object):
    def __init__(self, config, wordVec, vocabularyDic, trainData, validData, testData):
        self.config = config
        self.wordVec = wordVec
        self.vocabularyDic = vocabularyDic
        self.trainData = trainData
        self.validData = validData
        self.testData = testData
        self.build()




if __name__ == "__main__":
    return
