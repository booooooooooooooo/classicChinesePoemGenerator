#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

import os

from sets import Set
import time

#count the lines in each poem file
def analyzeRawData():
    for filePath in os.listdir(corpusdir):
        fin = open(corpusdir + filePath)
        lines = fin.readlines()
        print "****{:}".format(filePath)
        print len(lines)
        # print lines[0]
        # print lines[1]
        '''
        ****ming.all
        21716
        ****qing.all
        2217
        ****qsc_tab.txt
        18986
        ****qss_tab.txt
        183883
        ****qtais_tab.txt
        4650
        ****qts_tab.txt
        42974
        ****yuan.all
        10473
        '''

def prepareNPLMData(WINDOW_SIZE):
    print "***Start preparing data for NPLM ........"
    start = time.time()

    ENCODE = 'utf-8'
    corpusdir = "./data/raw_std_poem_all_from_rnnpg_data_emnlp-2014/"
    LINE_START = "<"
    LINE_END = ">"

    print "Making vocabularyDic........"
    corpus = []
    corpus.append(LINE_START)
    corpus.append(LINE_END)
    for filePath in os.listdir(corpusdir):
        print "===processing file {:}".format(filePath)
        fin = open(corpusdir + filePath)
        charList = fin.read().decode(ENCODE).split()# only chinese characters
        corpus += charList
    wordList = list( Set( corpus ) )
    # print len(wordList)
    # print wordList[0]
    # print wordList[1]
    """12174 + 2"""
    vocabularyDic = {}
    for i in xrange(len(wordList)):
        vocabularyDic[wordList[i]] = i
    # for k, v in vocabulary.items():
    #     print k
    #     print v


    print "Making window data........"
    windowData = []
    for filePath in os.listdir(corpusdir):
        print "===processing file {:}".format(filePath)
        fin = open(corpusdir + filePath)
        lines = fin.readlines()
        for line in lines:
            cleanedLine = line.decode(ENCODE).split() # decode and delete space and eol
            for i in xrange(len(cleanedLine)):
                center = vocabularyDic[ cleanedLine[i] ]
                context = []
                for j in range(i - WINDOW_SIZE, i + WINDOW_SIZE + 1):
                    if j != i:
                        if j < 0:
                            context.append(vocabularyDic[LINE_START])
                        elif j >= len(cleanedLine):
                            context.append(vocabularyDic[LINE_END])
                        else:
                            context.append(vocabularyDic[cleanedLine[j]])
                windowData.append((center, context))
    """15796478"""
    # print len(windowData)
    # print windowData[0][0]
    # print windowData[0][1][0]
    # print windowData[0][1][1]


    end = time.time()
    print "***Finish preparing data for NPLM. Time cost {:}".format(end - start)

    return vocabularyDic, windowData

if __name__ == "__main__":
    # prepareNPLMData(3)
