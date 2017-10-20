import tensorflow as tf
import os
from sets import Set


class NPLMADT:
    def __init__(self, ENCODE = 'utf-8', WINDOW_SIZE = 4, corpusdir = './utils/q5/', trainPath = './utils/q5/qtrain', validPath = './utils/q5/qvalid', testPath = './utils/q5/qtest'):
        self.ENCODE = ENCODE
        self.WINDOW_SIZE = WINDOW_SIZE
        self.corpusdir = corpusdir
        self.trainPath = trainPath
        self.validPath = validPath
        self.testPath = testPath

        self.wordList = self.makeWordList(ENCODE, corpusdir)
        self.trainData = self.makeWindowData(ENCODE, trainPath,WINDOW_SIZE, self.wordList)
        self.validData = self.makeWindowData(ENCODE, validPath,WINDOW_SIZE, self.wordList)
        self.testData = self.makeWindowData(ENCODE, testPath, WINDOW_SIZE,self.wordList)

    def makeWordList(self, ENCODE, corpusdir):
        corpus = []
        for filePath in os.listdir(corpusdir):
            fin = open(corpusdir + filePath)
            text = fin.read().decode(ENCODE)
            batch = list(text)
            corpus += batch
        wordList = list( Set( corpus ) )
        #TODO: \n is in wordList
        return wordList


    def makeWindowData(self, ENCODE, path, WINDOW_SIZE, wordList):
        inputData = []
        label = []
        fin = open(path)
        charList = list(fin.read().decode(ENCODE))
        for i in range(0, len(charList) - WINDOW_SIZE):
            wc = charList[i : i + WINDOW_SIZE / 2] +  charList[i + WINDOW_SIZE / 2 + 1 : i + WINDOW_SIZE]
            wp = charList[i + WINDOW_SIZE / 2]
            inputData.append( [wordList.index(w) for w in wc])
            label.append( wordList.index( wp ) )
        fin.close()
        return inputData, label

    def getV(self):
        return len(self.wordList)
    def getWordList(self):
        return self.wordList
    def getTrainData(self):
        return self.trainData
    def getValidData(self):
        return self.validData
    def getTestData(self):
        return self.testData




def sanity_check():
    adt = NPLMADT()
    trainInput, trainLabel = adt.getTrainData()
    validInput, validLabel = adt.getValidData()
    testInput, testLabel = adt.getTestData()
    print adt.getV()
    print len(trainInput), len(trainLabel)
    print len(validInput), len(validLabel)
    print len(testInput), len(testLabel)

if __name__ == "__main__":
    sanity_check()
