#encoding=utf-8
import tensorflow as tf
import sys
import os
from sets import Set
import numpy as np
import time

from utilData import UtilData
from utilGeneral import get_minibatches
from utilGeneral import getCosineSimilarities


class Config(object):
    def __init__(self, dim, WINDOW_SIZE, lr, n_epochs,batch_size, fileToSaveWordVectors, dirToSaveModel, dirToLog):
        self.lr = lr
        self.dim = dim
        self.WINDOW_SIZE = WINDOW_SIZE
        self.batch_size  = batch_size
        self.n_epochs = n_epochs

        self.fileToSaveWordVectors = fileToSaveWordVectors
        self.dirToSaveModel = dirToSaveModel
        self.dirToLog = dirToLog
    def getStringOfParas(self):
        return "dim_{:}_WINDOW_SIZE_{:}_lr_{:}_n_epochs_{:}_batch_size_{:}".format(self.dim, self.WINDOW_SIZE, self.lr, self.n_epochs, self.batch_size)


class SkipGram(object):
    def addPlaceHolder(self):
        self.cr = tf.placeholder(tf.int32, shape = [None, ], name = "centerWord")
        self.cx = tf.placeholder(tf.int32, shape =  [None,] , name = "contextWord")
    def addVariable(self):
        xavier_initializer = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope("prjectionLayer"):
            self.C = tf.get_variable("C", initializer = xavier_initializer, shape =  [len(self.vocabularyDic), self.config.dim], dtype = tf.float32)
        with tf.variable_scope("outputLayer"):
            self.W = tf.get_variable("W", initializer = xavier_initializer, shape = [self.config.dim, len(self.vocabularyDic)], dtype = tf.float32)
            self.b = tf.get_variable("b", initializer = xavier_initializer, shape = [len(self.vocabularyDic),], dtype = tf.float32)

    def getTrainLossFunc(self):
        x = tf.nn.embedding_lookup(self.C, self.cr)#batch_size * dim
        loss = tf.reduce_mean( tf.nn.nce_loss(weights=tf.transpose(self.W),
                                              biases=self.b,
                                              labels= tf.reshape(self.cx, [-1, 1]),
                                              inputs=x,
                                              num_sampled = 20,
                                              num_classes=len(self.vocabularyDic) ))
        return loss
    def getPredLossFunc(self):
        x = tf.nn.embedding_lookup(self.C, self.cr)#batch_size * dim
        logits = self.b + tf.matmul(x, self.W)#batch size * V
        loss =  tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits(labels = self.cx, logits = logits) )
        return loss
    def getTrainOp(self, loss):
        return tf.train.GradientDescentOptimizer(self.config.lr).minimize(loss)

    def build(self):
        self.addPlaceHolder()
        self.addVariable()
        self.trainLossFunc = self.getTrainLossFunc()
        self.trainOp = self.getTrainOp(self.trainLossFunc)
        self.predLossFunc = self.getPredLossFunc()

    def predict(self, sess, data):
        inputData, label = data
        loss = sess.run(self.predLossFunc, {self.cr : inputData, self.cx : label} )
        return loss

    def train_batch(self, sess, data_batch):
        inputData_batch, label_batch = data_batch
        _ , loss = sess.run([self.trainOp, self.trainLossFunc], {self.cr : inputData_batch, self.cx : label_batch})
        return loss

    def run_epoch(self, sess, data):
        minibatches = get_minibatches(data, self.config.batch_size)
        loss = 0.0
        # print "Batch count {:}".format(len(minibatches))
        for i in xrange(len(minibatches)):
            # print "Batch {:}".format(i)
            loss += self.train_batch(sess, minibatches[i])
        loss /= len(minibatches)
        return loss

    def fit(self, sess):
        print "***Start training model!"
        corTrainLoss = None
        bestValidLoss = sys.float_info.max
        trainData = self.trainData
        validData = self.validData
        for epoch in xrange(self.config.n_epochs) :
            start = time.time()
            print "***********Fitting Epoch {:}****************".format(epoch)
            trainLoss = self.run_epoch(sess, trainData)
            validLoss = self.predict(sess, validData)
            print "Train Loss   {:} \nValid Loss   {:}".format(trainLoss, validLoss)
            if validLoss < bestValidLoss:
                corTrainLoss = trainLoss
                bestValidLoss = validLoss
                print "^_^ A better model found!"
                tf.train.Saver().save(sess, self.config.dirToSaveModel + self.config.getStringOfParas())
            end = time.time()
            print "Epoch {:} cost {:} seconds".format(epoch, end - start)
            '''
            Each epoch costs about 400 seconds on whole corpus. Yeah!
            '''
        print "***Finish training model!"



        print "***Summary of model and config"
        print "Best valid Loss   {:} ".format(bestValidLoss)
        print "Cooresponding Train Loss   {:} ".format(corTrainLoss)
        print "Parameters used :" + self.config.getStringOfParas()
        print "Recovering best model ......"
        tf.train.Saver().restore(sess, self.config.dirToSaveModel + self.config.getStringOfParas())
        print "Word feature vectors saved in" + self.config.fileToSaveWordVectors
        fout = open(self.config.fileToSaveWordVectors, "w")
        np.save(fout, sess.run(self.C))
        fout.close()


    def intrinsicEvaluationCrossEntropy(self, sess):
        print "***Start CrossEntropy evaluation......"
        print "Recovering best model......"
        tf.train.Saver().restore(sess, self.config.dirToSaveModel + self.config.getStringOfParas())
        testLoss = self.predict(sess, self.testData)
        print "Test Loss   {:} ".format(testLoss)
        print "***Finish CrossEntropy evaluation. "
    def intrinsicEvaluationCosineSimilarity(self):
        print "***Start cosine similarity evaluation......"
        similarities = getCosineSimilarities(self.config.fileToSaveWordVectors, self.utilData.getMostFrequentChars, self.vocabularyDic)
        for i in xrange(len(similarities)):
            sim = similarities[i]
            print sim[0], sim[1], sim[2]
        print "***Finish cosine similarity evaluation...... "

    def __init__(self, config, utilData, vocabularyDic, trainData, validData, testData):
        self.config = config
        self.utilData = utilData
        self.vocabularyDic = vocabularyDic
        self.trainData = trainData
        self.validData = validData
        self.testData = testData
        self.build()


def sanityConfig():
    config = Config(30, 1, 0.1, 10, 500, "haha" , "haha", "hahaha" )
    print config.getStringOfParas()

def sanity_SkipGram():
    config = Config(30, 1, 0.1, 5, 500, "./output/skipGramWordVec" , "./saved_tf_model/", "./log_for_tensor_board/" )
    utilData = UtilData()
    vocabularyDic, trainData, validData, testData = utilData.prepareSkipGramData(config.WINDOW_SIZE, useSanityCorpus = False)
    with tf.Graph().as_default():
        model = SkipGram(config,utilData, vocabularyDic, trainData, validData, testData)
        with tf.Session() as session:
            # writer = tf.summary.FileWriter(config.dirToLog, session.graph)
            session.run( tf.global_variables_initializer() )
            model.fit(session)
            model.intrinsicEvaluationCrossEntropy(session)
            model.intrinsicEvaluationCosineSimilarity()
            # writer.close()
def evaluate():
    num = 100
    
    utilData = UtilData()
    similarities = getCosineSimilarities("./wordVec/wordVecSkipGram", utilData.getMostFrequentChars(num), utilData.prepareVocabularyDic())
    for sim in similarities:
        print sim[0].encode('utf-8'),
        print sim[1].encode('utf-8'),
        print sim[2]

def tune():
    print "Parameters tuning will be done after speed problem is solved!"
    #TODO


if __name__ == "__main__":
    # sanityConfig()
    sanity_SkipGram()
