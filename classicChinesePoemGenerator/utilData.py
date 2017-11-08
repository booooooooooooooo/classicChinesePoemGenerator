#!/usr/bin/python2.7
#encoding=utf-8

import os

from sets import Set
import time
from collections import Counter

class UtilData(object):
    def __init__(self):
        self.corpusDir = "./data/corpus/"
        self.ENCODE = 'utf-8'
        self.LINE_START = '<'
        self.LINE_END = '>'


    def analyzeCorpus(self):
        totalPoems = 0
        for filePath in os.listdir(self.corpusDir):
            if filePath.endswith('.txt') or filePath.endswith('.all'):
                fin = open(self.corpusDir + filePath)
                lines = fin.readlines()
                totalPoems += len(lines)
                print "****{:10d} poems in {:}".format(len(lines), filePath)
        print "****{:10d} poems in total".format(totalPoems)
    def getMostFrequentChars(self, num):
        cdir = self.corpusDir
        corpus = []
        corpus.append(self.LINE_START)
        corpus.append(self.LINE_END)
        for filePath in os.listdir(cdir):
            if filePath.endswith('.txt') or filePath.endswith('.all'):
                print "processing file {:}".format(filePath)
                fin = open(cdir + filePath)
                charList = fin.read().decode(self.ENCODE).split()# only chinese characters
                corpus += charList

        tupleList = Counter(corpus).most_common(num)
        return [t[0] for t in tupleList]

    def prepareVocabularyDic(self):
        cdir = self.corpusDir

        print "Start preparing vocabularyDic........"



        corpus = []
        corpus.append(self.LINE_START)
        corpus.append(self.LINE_END)
        for filePath in os.listdir(cdir):
            if filePath.endswith('.txt') or filePath.endswith('.all'):
                print "processing file {:}".format(filePath)
                fin = open(cdir + filePath)
                charList = fin.read().decode(self.ENCODE).split()# only chinese characters
                corpus += charList
        wordList = list( Set( corpus ) )
        vocabularyDic = {}
        for i in xrange(len(wordList)):
            vocabularyDic[wordList[i]] = i
        print "****Vocabulary size : {:d}".format(len(vocabularyDic))
        print "Finish preparing vocabularyDic........"
        return vocabularyDic

    def prepareNPLMData(self, WINDOW_SIZE):

        cdir = self.corpusDir

        print "Start preparing data for NPLM ........"
        print "Start timer"
        start = time.time()
        vocabularyDic = self.prepareVocabularyDic(useSanityCorpus)
        print "Making input-label pairs........"
        windowData = []
        for filePath in os.listdir(cdir):
            if filePath.endswith('.txt') or filePath.endswith('.all'):
                print "processing file {:}".format(filePath)
                fin = open(cdir + filePath)
                lines = fin.readlines()
                for line in lines:
                    cleanedLine = line.decode(self.ENCODE).split() # decode and delete space and eol
                    for i in xrange(len(cleanedLine)):
                        center = vocabularyDic[ cleanedLine[i] ]
                        context = []
                        for j in range(i - WINDOW_SIZE, i + WINDOW_SIZE + 1):
                            if j != i:
                                if j < 0:
                                    context.append(vocabularyDic[self.LINE_START])
                                elif j >= len(cleanedLine):
                                    context.append(vocabularyDic[self.LINE_END])
                                else:
                                    context.append(vocabularyDic[cleanedLine[j]])
                        windowData.append((context, center))
        trainData = zip(*windowData[0 : len(windowData) - 2000])#(inputs, labels)
        validData = zip(*windowData[len(windowData) - 2000 : len(windowData) - 1000])#(inputs, labels)
        testData = zip(*windowData[len(windowData) - 1000 : len(windowData)])#(inputs, labels)

        end = time.time()
        print "****{:d} datum in total".format(len(windowData))
        print "****Time cost {:} seconds".format(end - start)
        print "Finish preparing data for NPLM"

        return vocabularyDic, trainData, validData, testData

    def prepareSkipGramData(self, WINDOW_SIZE):
        cdir = self.corpusDir
        if useSanityCorpus:
            cdir = self.corpusDirSanity

        print "Start preparing data for SkipGram ........"
        print "Start timer"
        start = time.time()
        vocabularyDic = self.prepareVocabularyDic(useSanityCorpus)
        print "Making input-label pairs........"
        windowData = []
        for filePath in os.listdir(cdir):
            if filePath.endswith('.txt') or filePath.endswith('.all'):
                print "processing file {:}".format(filePath)
                fin = open(cdir + filePath)
                lines = fin.readlines()
                for line in lines:
                    cleanedLine = line.decode(self.ENCODE).split() # decode and delete space and eol
                    for i in xrange(len(cleanedLine)):
                        center = vocabularyDic[ cleanedLine[i] ]
                        for j in range(i - WINDOW_SIZE, i + WINDOW_SIZE + 1):
                            if j != i:
                                if j < 0:
                                    windowData.append((center, vocabularyDic[self.LINE_START]))
                                elif j >= len(cleanedLine):
                                    windowData.append(( center, vocabularyDic[self.LINE_END]))
                                else:
                                    windowData.append((center, vocabularyDic[cleanedLine[j]]))
        trainData = zip(*windowData[0 : len(windowData) - 2000])#(inputs, labels)
        validData = zip(*windowData[len(windowData) - 2000 : len(windowData) - 1000])#(inputs, labels)
        testData = zip(*windowData[len(windowData) - 1000 : len(windowData)])#(inputs, labels)


        end = time.time()
        print "****{:d} datum in total".format(len(windowData))
        print "****Time cost {:} seconds".format(end - start)
        print "Finish preparing data for SkipGram"

        return vocabularyDic, trainData, validData, testData
    def prepareQuatrain5Data(self):
        cdir = self.corpusDir

        vocabularyDic = self.prepareVocabularyDic(useSanityCorpus)
        allData = []
        for filePath in os.listdir(cdir):
            if filePath.endswith('.txt') or filePath.endswith('.all'):
                print "processing file {:}".format(filePath)
                fin = open(cdir + filePath)
                lines = fin.readlines()
                for line in lines:
                    cleanedLine = line.decode(self.ENCODE).split()
                    if len(cleanedLine) == 20:
                        allData += prepareQuatrain5DataHelper(self, cleanedLine)

        trainData = zip(*allData[0 : len(allData) - 2000])#(inputs, labels)
        validData = zip(*allData[len(allData) - 2000 : len(allData) - 1000])#(inputs, labels)
        testData = zip(*allData[len(allData) - 1000 : len(allData)])#(inputs, labels)

        return vocabularyDic, trainData, validData, testData
    def prepareQuatrain5DataHelper(self, cleanedLine):
        result = []
        for i in [0, 5, 10, 15]:
            result.append( (cleanedLine[i], cleanedLine[i : i + 5]) )
        return result

if __name__ == "__main__":
    utilData = UtilData()
    # utilData.analyzeCorpus()
    # utilData.prepareVocabularyDic()
    # utilData.prepareNPLMData(1)
    # utilData.prepareSkipGramData(1)
    # vocabularyDic, trainData, validData, testData = utilData.prepareQuatrain5Data()
    # print len(vocabularyDic)
    # print trainData[0][0], trainData[1][0]
    #
    # print validData[0][0], validData[1][0]
    # print testData[0][0], testData[1][0]
    chars = utilData.getMostFrequentChars(100)
    for c in chars:
        print c,
