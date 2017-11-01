#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

import os

from sets import Set
import time


ENCODE = 'utf-8'
corpusdir = "./data/raw_std_poem_all_from_rnnpg_data_emnlp-2014/"
WINDOW_SIZE = 1 # 2 * WINDOW_SIZE contextWords, and one center word

LINE_START = "<"
LINE_END = ">"



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




def makeVolDic():
    corpus = []
    for filePath in os.listdir(corpusdir):
        fin = open(corpusdir + filePath)
        charList = fin.read().decode(ENCODE).split()# only chinese characters
        corpus += charList
    corpus.append(LINE_START)
    corpus.append(LINE_END)
    wordList = list( Set( corpus ) )
    print len(wordList)
    # print wordList[0]
    # print wordList[1]
    """
    12174 + 2
    Vocabulary size
    Vocabulary includes LINE_START and LINE_END
    """


def makeWindowData():
    start = time.time()
    windowData = []
    for filePath in os.listdir(corpusdir):
        print "===processing file {:}".format(filePath)
        fin = open(corpusdir + filePath)
        lines = fin.readlines()
        for line in lines:
            cleanedLine = line.decode(ENCODE).split() # decode and delete space and eol
            for i in xrange(len(cleanedLine)):
                center = cleanedLine[i]
                context = []
                for j in range(i - WINDOW_SIZE, i + WINDOW_SIZE + 1):
                    if j != i:
                        if j < 0:
                            context.append(LINE_START)
                        elif j >= len(cleanedLine):
                            context.append(LINE_END)
                        else:
                            context.append(cleanedLine[j])
                windowData.append((center, context))
    print len(windowData)
    """15796478"""
    # print windowData[0][0]
    # print windowData[0][1][0]
    # print windowData[0][1][1]
    end = time.time()
    print "Time cost {:}".format(end - start)


if __name__ == "__main__":
    makeWindowData()
