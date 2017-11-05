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


class Generator_RNN_ODM(object):
    def __init__(self):
        self.vocabularyDic, self.embeddingMatrix, self.allData = prepare_Generator_RNN_Data()#list of tuple
        self.trainData = self.allData[0 : len(self.allData) - 2000]#list of tuple (center word, [context1, context2...])
        self.validData = self.allData[len(self.allData) - 2000 : len(self.allData) - 1000]#list of tuple
        self.testData = self.allData[len(self.allData) - 1000 : len(self.allData)]#list of tuple

    def getVocabularySize(self):
        return len(self.vocabularyDic)
    def getVocabularyDic(self):
        return self.vocabularyDic
    def getRandomChars(self, n_chars):
        chars = self.vocabularyDic.keys()
        indices = np.arange(len(chars))
        np.random.shuffle(indices)
        return [chars[indices[i]] for i in xrange(n_chars)]
    def getAllData(self):
        return zip(*self.allData)
    def getTrainData(self):
        return zip(*self.trainData)
    def getValidData(self):
        return zip(*self.validData)
    def getTestData(self):
        return zip(*self.testData)



class Generator_RNN(object):
    #TODO



def sanity_NPLMODM():
    print "In process!"
    #TODO

def sanity_NPLM():
    print "In process!"
    #TODO


def tune():
    print "Parameters tuning will be done after speed problem is solved!"
    #TODO



if __name__ == "__main__":
    sanity_NPLMODM()
    # sanity_NPLM()
    # tune()
