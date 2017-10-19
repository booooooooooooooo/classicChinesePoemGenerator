import tensorflow as tf
from nltk.corpus.reader.plaintext import PlaintextCorpusReader

class DataToLearnWFV:
    def __init__(self, ENCODE = 'utf-8', WINDOW_SIZE = 4, corpusdir = '/Users/bo/Documents/297And8/chinesePeotryGenerater_deliverable2And3/utils/data/rnnpg_data_emnlp-2014/partitions_in_Table_2/rnnpg/', trainPath = '/Users/bo/Documents/297And8/chinesePeotryGenerater_deliverable2And3/utils/data/rnnpg_data_emnlp-2014/partitions_in_Table_2/rnnpg/qtrain', validPath = '/Users/bo/Documents/297And8/chinesePeotryGenerater_deliverable2And3/utils/data/rnnpg_data_emnlp-2014/partitions_in_Table_2/rnnpg/qvalid', testPath = '/Users/bo/Documents/297And8/chinesePeotryGenerater_deliverable2And3/utils/data/rnnpg_data_emnlp-2014/partitions_in_Table_2/rnnpg/qtest'):
        self.ENCODE = ENCODE
        self.WINDOW_SIZE = WINDOW_SIZE
        self.corpusdir = corpusdir
        self.trainPath = trainPath
        self.validPath = validPath
        self.testPath = testPath

        corpus = PlaintextCorpusReader(corpusdir, '.*').raw()
        self.wordList = ''.join(set( corpus.split() ))

    def getV(self):
        return len(self.wordList)

    def getWindowData(self, path):
        inputData = []
        label = []
        fin = open(path)
        charList = fin.read().decode(self.ENCODE).split()
        for i in range(0, len(charList) - self.WINDOW_SIZE):
            wc = charList[i : i + self.WINDOW_SIZE / 2] +  charList[i + self.WINDOW_SIZE / 2 + 1 : i + self.WINDOW_SIZE]
            wp = charList[i + self.WINDOW_SIZE / 2]
            inputData.append( [self.wordList.index(w) for w in wc])
            label.append( self.wordList.index( wp ) )
            # print newLabel

        fin.close()
        #TODO: change to real size
        return inputData[0 : 100], label[0:100]

    def getTrain(self):
        return self.getWindowData(self.trainPath)

    def getValid(self):
        return self.getWindowData(self.validPath)

    def getTest(self):
        return self.getWindowData(self.testPath)






def sanity_check():
    data = DataToLearnWFV()
    trainInput, trainLabel  = data.getTrain()
    print len(trainInput), len(trainLabel)
    # for i in xrange(len(trainInput)):
    #     print trainInput[i], trainLabel[i]


if __name__ == "__main__":
    sanity_check()
