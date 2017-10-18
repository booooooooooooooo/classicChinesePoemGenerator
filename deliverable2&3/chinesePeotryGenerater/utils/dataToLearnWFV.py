from nltk.corpus.reader.plaintext import PlaintextCorpusReader

class DataToLearnWFV:
    def __init__(self, ENCODE = 'utf-8', WINDOW_SIZE = 4, corpusdir = './data/rnnpg_data_emnlp-2014/partitions_in_Table_2/rnnpg/', trainPath = './data/rnnpg_data_emnlp-2014/partitions_in_Table_2/rnnpg/qtrain', validPath = './data/rnnpg_data_emnlp-2014/partitions_in_Table_2/rnnpg/qvalid', testPath = './data/rnnpg_data_emnlp-2014/partitions_in_Table_2/rnnpg/qtest'):
        self.ENCODE = ENCODE
        self.WINDOW_SIZE = WINDOW_SIZE
        self.corpusdir = corpusdir
        self.trainPath = trainPath
        self.validPath = validPath
        self.testPath = testPath

        corpus = PlaintextCorpusReader(corpusdir, '.*').raw()
        self.wordList = ''.join(set( corpus.split() ))

    def getWordList(self):
        return self.wordList

    def getV(self):
        return len(self.wordList)

    def getWindowData(self, path):
        data = {}
        fin = open(path)
        charList = fin.read().decode(self.ENCODE).split()
        for i in range(0, len(charList) - self.WINDOW_SIZE):
            data[charList[i + self.WINDOW_SIZE / 2]] = charList[i : i + self.WINDOW_SIZE / 2] + charList[i + self.WINDOW_SIZE / 2 + 1 : i + self.WINDOW_SIZE]
        fin.close()
        return data

    def getTrain(self):
        return self.getWindowData(self.trainPath)

    def getValid(self):
        return self.getWindowData(self.validPath)

    def getTest(self):
        return self.getWindowData(self.testPath)






def sanity_check():
    data = DataToLearnWFV()
    for key in data.getTrain():
        print key


if __name__ == "__main__":
    sanity_check()