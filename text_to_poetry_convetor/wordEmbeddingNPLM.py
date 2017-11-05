import tensorflow as tf
import sys
import os
from sets import Set
import numpy as np
from scipy import spatial

from utilData import prepareNPLMData
from utilGeneral import get_minibatches
from utilGeneral import getRandomChars


class Config(object):
    def __init__(self, lr, dim, h, WINDOW_SIZE, n_epochs,batch_size, fileToSaveWordVectors, dirToSaveModel, dirToLog):
        self.lr = lr
        self.dim = dim
        self.h = h
        self.WINDOW_SIZE = WINDOW_SIZE
        self.batch_size  = batch_size
        self.n_epochs = n_epochs

        self.fileToSaveWordVectors = fileToSaveWordVectors
        self.dirToSaveModel = dirToSaveModel
        self.dirToLog = dirToLog
    def getStringOfParas(self):
        #TODO
        return "Mohahahaha"

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
                tf.train.Saver().save(sess, self.config.dirToSaveModel + self.config.getStringOfParas())
        print "***Finish training model!"

        tf.train.Saver().restore(sess, self.config.dirToSaveModel + self.config.getStringOfParas())
        np.save(self.config.fileToSaveWordVectors, sess.run(self.C))
        testData = self.odm.getTestData()
        corTestLoss = self.predict(sess, testData)
        print "***Summary of model and config"
        print "Best valid Loss   {:} ".format(bestValidLoss)
        print "Cooresponding Train Loss   {:} ".format(corTrainLoss)
        print "Cooresponding Test Loss   {:} ".format(corTestLoss)
        print "Parameters used :" + self.config.getStringOfParas()
        print "Word feature vectors saved in" + self.config.fileToSaveWordVectors


    def intrinsicEvaluation(self, sess):
        tf.train.Saver().restore(sess, self.config.dirToSaveModel + self.config.getStringOfParas())
        wordFeatureVectors = np.load(self.config.fileToSaveWordVectors)
        print wordFeatureVectors
        print "***Start intrinsic evaluation......"
        n_chars = 1000
        chars = utilGeneral.getRandomChars(n_chars, vocabularyDic)
        similarities = []
        for i in xrange(n_chars):
            for j in xrange(n_chars):
                if j > i:
                    a, b = wordFeatureVectors[ self.odm.getVocabularyDic()[chars[i]] ], wordFeatureVectors[ self.odm.getVocabularyDic()[chars[j]] ]
                    score = (np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b)))
                    similarities.append((chars[i], chars[j], score))
        similarities = sorted(similarities, key=lambda sim: sim[2], reverse=True)
        for i in xrange(50):
            sim = similarities[i]
            print sim[0], sim[1], sim[2]

        print "***Finish intrinsic evaluation. "


    def __init__(self, config, vocabularyDic, trainData, validData, testData):
        self.config = config
        self.vocabularyDic = vocabularyDic
        self.trainData = trainData
        self.validData = validData
        self.build()



def sanity_NPLM():
    config = Config(lr = 0.5,  dim = 30, h = 50, WINDOW_SIZE = 1, n_epochs = 20, batch_size=50, "./data/wordFeatureVector.txt" , "./saved_tf_model/", "./log_for_tensor_board" )
    vocabularyDic, trainData, validData, testData = UtilData().prepareNPLMData(config.WINDOW_SIZE)
    with tf.Graph().as_default():
        model = NPLM(config,vocabularyDic, trainData, validData, testData)
        with tf.Session() as sess:
            # writer = tf.summary.FileWriter(config.dirToLog, session.graph)
            sess.run( tf.global_variables_initializer() )
            model.fit(sess)
            model.intrinsicEvaluation(sess)
            # writer.close()


def tune():
    print "Parameters tuning will be done after speed problem is solved!"
    #TODO



if __name__ == "__main__":
    sanity_NPLM()
    # tune()
