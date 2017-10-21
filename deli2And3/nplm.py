import tensorflow as tf
import sys
import os
from sets import Set


class Config(object):
    def __init__(self, lr, dim, h, WINDOW_SIZE, dirToSaveModel, dirToLog):
        self.lr = lr
        self.dim = dim
        self.h = h
        self.WINDOW_SIZE = WINDOW_SIZE
        self.pathToSaveModel = dirToSaveModel
        self.dirToLog = dirToLog


class NPLMODM(object):
    def __init__(self, WINDOW_SIZE = 4, proportion = 0.1, ENCODE = 'utf-8', corpusdir = './utils/q5/', trainPath = './utils/q5/qtrain', validPath = './utils/q5/qvalid', testPath = './utils/q5/qtest'):
        self.ENCODE = ENCODE
        self.WINDOW_SIZE = WINDOW_SIZE
        self.corpusdir = corpusdir
        self.trainPath = trainPath
        self.validPath = validPath
        self.testPath = testPath

        self.wordList = self.makeWordList(ENCODE, corpusdir, proportion)
        self.trainData = self.makeWindowData(ENCODE, trainPath,WINDOW_SIZE, self.wordList, proportion)
        self.validData = self.makeWindowData(ENCODE, validPath,WINDOW_SIZE, self.wordList, proportion)
        self.testData = self.makeWindowData(ENCODE, testPath, WINDOW_SIZE,self.wordList, proportion)

    def makeWordList(self, ENCODE, corpusdir, proportion):
        corpus = []
        for filePath in os.listdir(corpusdir):
            fin = open(corpusdir + filePath)
            charList = list(fin.read().decode(ENCODE))
            charList = charList[0 : int(proportion * len(charList) )]
            corpus += charList
        wordList = list( Set( corpus ) )
        #TODO: \n or blank are in wordList; other punctuations are not in wordList
        return wordList


    def makeWindowData(self, ENCODE, path, WINDOW_SIZE, wordList, proportion):
        inputData = []
        label = []
        fin = open(path)
        charList = list(fin.read().decode(ENCODE))
        charList = charList[0 : int(proportion * len(charList) )]
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



class NPLM(object):
    def addPlaceHolder(self):
        self.cw = tf.placeholder(tf.int32, shape = [None, self.config.WINDOW_SIZE - 1], name = "contextWords")
        self.pw = tf.placeholder(tf.int32, shape =  [None] , name = "predictedWord")
    def addVariable(self):
        with tf.variable_scope("embeddingLayer"):
            self.C = tf.get_variable("C", [self.odm.getV(), self.config.dim])
        with tf.variable_scope("hiddenLayer"):
            self.H = tf.get_variable("H", [self.config.dim * (self.config.WINDOW_SIZE - 1), self.config.h])
            self.d = tf.get_variable("d", [self.config.h])
        with tf.variable_scope("outputLayer"):
            self.U = tf.get_variable("U", [self.config.h, self.odm.getV()])
            self.b = tf.get_variable("b", [self.odm.getV()])

    def getLossFunc(self):
        x = tf.reshape( tf.nn.embedding_lookup(self.C, self.cw), shape = [-1, self.config.dim * (self.config.WINDOW_SIZE - 1)] ) #  batch size * (dim*(WINDOW_SIZE - 1) )
        z1 = self.d + tf.matmul(x, self.H)#  batch size * h
        h1 = tf.tanh(z1)#  batch size * h
        logits = self.b + tf.matmul(h1, self.U)#  batch size * V

        return tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits(labels = self.pw, logits = logits) )

    def getTrainOp(self, loss):
        return tf.train.GradientDescentOptimizer(self.config.lr).minimize(loss)

    def build(self):
        self.addPlaceHolder()
        self.addVariable()
        self.lossFunc = self.getLossFunc()
        self.trainOp = self.getTrainOp(self.lossFunc)

    def train(self, sess, inputData, label):
        _ , loss = sess.run([self.trainOp, self.lossFunc], {self.cw : inputData, self.pw : label})
        return loss

    def predict(self, sess, inputData, label):
        loss = sess.run(self.lossFunc, {self.cw : inputData, self.pw : label} )
        return loss

    def fit(self, sess):
        trainLoss = sys.float_info.max
        validLoss = sys.float_info.max

        for epoch in xrange(10000) :
            print "***********Fitting Epoch {:}****************".format(epoch)
            trainInput, trainLabel = self.odm.getTrainData()
            validInput, validLabel = self.odm.getValidData()
            trainLoss = self.train(sess, trainInput, trainLabel)
            validLoss = self.predict(sess, validInput, validLabel)
            print "Train Loss   {:} \nValid Loss   {:}".format(trainLoss, validLoss)

        print "*********Training finished!*********"

        print "*********Testing model***********"
        testInput, testLabel = self.odm.getTestData()
        testLoss = self.predict(sess, testInput, testLabel)
        print "Test Loss   {:} ".format(testLoss)

        print "*********Summary********************"
        print self.C
        print self.config
        print trainLoss
        print validLoss
        print testLoss

        return trainLoss, validLoss, testLoss

    def __init__(self, config, odm):
        self.config = config
        self.odm = odm
        self.build()




def sanity_NPLMADT():
    adt = NPLMADT()
    trainInput, trainLabel = adt.getTrainData()
    validInput, validLabel = adt.getValidData()
    testInput, testLabel = adt.getTestData()
    print adt.getV()
    print len(trainInput), len(trainLabel)
    print len(validInput), len(validLabel)
    print len(testInput), len(testLabel)

def sanity_NPLM():
    config = Config(10, 5, 3, 6,  "./model/", "./log" )
    odm = NPLMODM(WINDOW_SIZE = config.WINDOW_SIZE, proportion = 0.1)
    with tf.Graph().as_default():
        model = NPLM(config, odm)
        with tf.Session() as session:
            # writer = tf.summary.FileWriter(config.dirToLog, session.graph)
            session.run( tf.global_variables_initializer() )
            trainLoss, validLoss, testLoss = model.fit(session)
            # writer.close()

#
# def tune():
#     #TODO: how to tune????????? 



if __name__ == "__main__":
    # sanity_NPLMADT()
    sanity_NPLM()
