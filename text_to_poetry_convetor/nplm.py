import tensorflow as tf
import sys
import os
from sets import Set
import numpy as np

from processData import prepareNPLMData
from util import get_minibatches


class Config(object):
    def __init__(self, lr, dim, h, WINDOW_SIZE, n_epochs,batch_size, fileToSaveWordVectors, dirToSaveModel, dirToLog):
        self.lr = lr
        self.dim = dim
        self.h = h
        self.WINDOW_SIZE = WINDOW_SIZE
        self.batch_size  = batch_size
        self.n_epochs = n_epochs

        self.fileToSaveWordVectors = fileToSaveWordVectors
        self.pathToSaveModel = dirToSaveModel
        self.dirToLog = dirToLog
    def getStringOfParas(self):
        #TODO
        return "Mohahahaha"


class NPLMODM(object):
    def __init__(self, WINDOW_SIZE):
        self.WINDOW_SIZE = WINDOW_SIZE
        self.vocabularyDic, self.windowData = prepareNPLMData(WINDOW_SIZE)#list of tuple
        self.trainData = self.windowData[0 : len(self.windowData) - 2000]#list of tuple (center word, [context1, context2...])
        self.validData = self.windowData[len(self.windowData) - 2000 : len(self.windowData) - 1000]#list of tuple
        self.testData = self.windowData[len(self.windowData) - 1000 : len(self.windowData)]#list of tuple

    def getVocabularySize(self):
        return len(self.vocabularyDic)
    def getVocabularyDic(self):
        return self.vocabularyDic
    def getAllData(self):
        return zip(*self.windowData)
    def getTrainData(self):
        return zip(*self.trainData)
    def getValidData(self):
        return zip(*self.validData)
    def getTestData(self):
        return zip(*self.testData)



class NPLM(object):
    def addPlaceHolder(self):
        self.cw = tf.placeholder(tf.int32, shape = [None, self.config.WINDOW_SIZE * 2], name = "contextWords")
        self.pw = tf.placeholder(tf.int32, shape =  [None] , name = "predictedWord")
    def addVariable(self):
        with tf.variable_scope("embeddingLayer"):
            self.C = tf.get_variable("C", [self.odm.getVocabularySize(), self.config.dim])
        with tf.variable_scope("hiddenLayer"):
            self.H = tf.get_variable("H", [self.config.dim * (self.config.WINDOW_SIZE * 2), self.config.h])
            self.d = tf.get_variable("d", [self.config.h])
        with tf.variable_scope("outputLayer"):
            self.U = tf.get_variable("U", [self.config.h, self.odm.getVocabularySize()])
            self.b = tf.get_variable("b", [self.odm.getVocabularySize()])

    def getLossFunc(self):
        x = tf.reshape( tf.nn.embedding_lookup(self.C, self.cw), shape = [-1, self.config.dim * (self.config.WINDOW_SIZE * 2)] ) #  batch size * (dim*(WINDOW_SIZE - 1) )
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

    def predict(self, sess, data):
        inputData, label = data
        loss = sess.run(self.lossFunc, {self.cw : inputData, self.pw : label} )
        return loss

    def train_batch(self, sess, data_batch):
        inputData_batch, label_batch = data_batch
        _ , loss = sess.run([self.trainOp, self.lossFunc], {self.cw : inputData_batch, self.pw : label_batch})
        return loss

    def run_epoch(self, sess, data):
        minibatches = get_minibatches(data, self.config.batch_size)
        loss = 0.0
        for i in xrange(len(minibatches)):
            # print "Batch {:}".format(i)
            loss += self.train_batch(sess, minibatches[i])
        loss /= len(minibatches)
        return loss

    def fit(self, sess):
        print "***Start training model!"
        corTrainLoss = None
        bestValidLoss = sys.float_info.max
        trainData = self.odm.getTrainData()
        validData = self.odm.getValidData()
        for epoch in xrange(self.config.n_epochs) :
            print "***********Fitting Epoch {:}****************".format(epoch)
            trainLoss = self.run_epoch(sess, trainData)
            validLoss = self.predict(sess, validData)
            print "Train Loss   {:} \nValid Loss   {:}".format(trainLoss, validLoss)
            if validLoss < bestValidLoss:
                corTrainLoss = trainLoss
                bestValidLoss = validLoss
                print "^_^ A better model found!"
                tf.train.Saver().save(sess, self.config.pathToSaveModel + self.config.getStringOfParas())
        print "***Finish training model!"

        tf.train.Saver().restore(sess, self.config.pathToSaveModel + self.config.getStringOfParas())
        np.save(self.config.fileToSaveWordVectors, sess.run(self.C))
        print "***Summary of model and config"
        print "Best valid Loss   {:} ".format(bestValidLoss)
        print "Cooresponding Train Loss   {:} ".format(corTrainLoss)
        print "Parameters used :" + self.config.getStringOfParas()
        print "Word feature vectors saved in" + self.config.fileToSaveWordVectors
        wordFeatureVectors = np.load(self.config.fileToSaveWordVectors)
        print wordFeatureVectors

    def intrinsicEvaluation(self, sess):
        tf.train.Saver().restore(sess, self.config.pathToSaveModel + self.config.getStringOfParas())
        print "***Start intrinsic evaluation......"
        testData = self.odm.getTestData()
        corTestLoss = self.predict(sess, testData)
        print "***Finish intrinsic evaluation. Test loss {:}".format(corTestLoss)
    def extrinsicEvaluation(self, sess):
        tf.train.Saver().restore(sess, self.config.pathToSaveModel + self.config.getStringOfParas())
        print "***Start extrinsic evaluation......"
        #TODO: real case word similarity
        print "***Finish extrinsic evaluation"

    def __init__(self, config, odm):
        self.config = config
        self.odm = odm
        self.build()




def sanity_NPLMODM():
    odm = NPLMODM()
    trainInput, trainLabel = odm.getTrainData()
    validInput, validLabel = odm.getValidData()
    testInput, testLabel = odm.getTestData()
    print len(trainInput), len(trainLabel)
    print len(validInput), len(validLabel)
    print len(testInput), len(testLabel)

def sanity_NPLM():
    config = Config(10, 10, 10, 1, 0, 50, "./data/wordFeatureVector.txt" , "./saved_tf_model/", "./log_for_tensor_board" )
    odm = NPLMODM(WINDOW_SIZE = config.WINDOW_SIZE)
    with tf.Graph().as_default():
        model = NPLM(config, odm)
        with tf.Session() as session:
            # writer = tf.summary.FileWriter(config.dirToLog, session.graph)
            session.run( tf.global_variables_initializer() )
            model.fit(session)
            model.intrinsicEvaluation(session)
            model.extrinsicEvaluation(session)
            # writer.close()

#
# def tune():
#     #TODO: how to tune?????????



if __name__ == "__main__":
    # sanity_NPLMODM()
    sanity_NPLM()
