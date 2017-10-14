#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

ENCODE = 'utf-8'

class DataSet:
    def __init__(self, trainPath=None, validPath = None, testPath = None):
        if not trainPath:
            trainPath = "./data/boQuatrain5/qtrain"
        if not validPath:
            validPath = "./data/boQuatrain5/qvalid"
        if not testPath:
            testPath = "./data/boQuatrain5/qtest"
        self.trainPath = trainPath
        self.validPath = validPath
        self.testPath = testPath
    def getData(self, usage):
        fin = None
        if usage == "train":
            fin = open(self.trainPath)
        elif usage == "valid":
            fin = open(self.validPath)
        elif usage == "test":
            fin = open(self.testPath)
        else:
            raise ValueError("Wrong parameter!")

        lines = fin.read().decode(ENCODE).split('\n')
        inputData = []
        labelData = []
        for line in lines:
            if len(line) == 20:
                inputData.append(line[0:5])
                labelData.append(line[5:20])
        return inputData, labelData
    def getTrainData(self):
        return self.getData("train")

    def getValidData(self):
        return self.getData("valid")

    def getTestData(self):
        return self.getData("test")

def sanity_test():
    dataSet = DataSet()
    inputTrain, labelTrain = dataSet.getTestData()
    print len(inputTrain)
    print len(labelTrain)

if __name__ == "__main__":
    sanity_test()
