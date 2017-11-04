import tensorflow as tf
import sys
import os
from sets import Set
import numpy as np
import time

from data_util import prepareSkipGramData
from general_util import get_minibatches
from scipy import spatial


class Config(object):
    def __init__(self, lr, dim, WINDOW_SIZE, n_epochs,batch_size, fileToSaveWordVectors, dirToSaveModel, dirToLog):
        self.lr = lr
        self.dim = dim
        self.WINDOW_SIZE = WINDOW_SIZE
        self.batch_size  = batch_size
        self.n_epochs = n_epochs

        self.fileToSaveWordVectors = fileToSaveWordVectors
        self.pathToSaveModel = dirToSaveModel
        self.dirToLog = dirToLog
    def getStringOfParas(self):
        #TODO
        return "hahahahaha"


class SkipGramODM(object):
    def __init__(self, WINDOW_SIZE):
        self.WINDOW_SIZE = WINDOW_SIZE
        self.vocabularyDic, self.windowData = prepareSkipGramData(WINDOW_SIZE, useTinyCorpus = False)#list of tuple (center word, context word)
        self.trainData = self.windowData[0 : len(self.windowData) - 2000]#list of tuple
        self.validData = self.windowData[len(self.windowData) - 2000 : len(self.windowData) - 1000]#list of tuple
        self.testData = self.windowData[len(self.windowData) - 1000 : len(self.windowData)]#list of tuple

    def getVocabularySize(self):
        return len(self.vocabularyDic)
    def getVocabularyDic(self):
        return self.vocabularyDic
    def getRandomChars(self, n_chars):
        chars = self.vocabularyDic.keys()
        indices = np.arange(len(chars))
        np.random.shuffle(indices)
        return [chars[indices[i]] for i in xrange(n_chars)]
    def getAllData(self):
        return zip(*self.windowData)
    def getTrainData(self):
        return zip(*self.trainData)
    def getValidData(self):
        return zip(*self.validData)
    def getTestData(self):
        return zip(*self.testData)



class SkipGram(object):
    def addPlaceHolder(self):
        self.cr = tf.placeholder(tf.int32, shape = [None, ], name = "centerWord")
        self.cx = tf.placeholder(tf.int32, shape =  [None,] , name = "contextWord")
    def addVariable(self):
        xavier_initializer = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope("prjectionLayer"):
            self.C = tf.get_variable("C", initializer = xavier_initializer, shape =  [self.odm.getVocabularySize(), self.config.dim], dtype = tf.float32)
        with tf.variable_scope("outputLayer"):
            self.W = tf.get_variable("W", initializer = xavier_initializer, shape = [self.config.dim, self.odm.getVocabularySize()], dtype = tf.float32)
            self.b = tf.get_variable("b", initializer = xavier_initializer, shape = [self.odm.getVocabularySize(),], dtype = tf.float32)

    def getTrainLossFunc(self):
        x = tf.nn.embedding_lookup(self.C, self.cr)#batch_size * dim
        loss = tf.reduce_mean( tf.nn.nce_loss(weights=tf.transpose(self.W),
                                              biases=self.b,
                                              labels= tf.reshape(self.cx, [-1, 1]),
                                              inputs=x,
                                              num_sampled = 20,
                                              num_classes=self.odm.getVocabularySize()))
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
        trainData = self.odm.getTrainData()
        validData = self.odm.getValidData()
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
                tf.train.Saver().save(sess, self.config.pathToSaveModel + self.config.getStringOfParas())
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
        tf.train.Saver().restore(sess, self.config.pathToSaveModel + self.config.getStringOfParas())
        print "Word feature vectors saved in" + self.config.fileToSaveWordVectors
        fout = open(self.config.fileToSaveWordVectors, "w")
        np.save(fout, sess.run(self.C))
        fout.close()
        # print type(sess.run(self.C))
        # print sess.run(self.C)


    def intrinsicEvaluation(self, sess):
        print "***Start intrinsic evaluation......"
        print "Recovering best model......"
        tf.train.Saver().restore(sess, self.config.pathToSaveModel + self.config.getStringOfParas())
        testLoss = self.predict(sess, self.odm.getTestData())
        print "Test Loss   {:} ".format(testLoss)
        print "Loading word vectors......"
        fin = open(self.config.fileToSaveWordVectors)
        wordFeatureVectors = np.load(fin)
        fin.close()
        print wordFeatureVectors
        print "Some character cosine similirities......"
        n_chars = 1000
        chars = self.odm.getRandomChars(n_chars)
        similarities = []
        for i in xrange(n_chars):
            for j in xrange(n_chars):
                if j > i:
                    a, b = wordFeatureVectors[ self.odm.getVocabularyDic()[chars[i]] ], wordFeatureVectors[ self.odm.getVocabularyDic()[chars[j]] ]
                    score = (np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b)))
                    similarities.append((chars[i], chars[j], score))
        similarities = sorted(similarities, key=lambda sim: sim[2], reverse=True)
        for i in xrange(len(similarities)):
            sim = similarities[i]
            # if sim < 0.9 and sim > 0.89:
            #     print sim[0], sim[1], sim[2]
            print sim[0], sim[1], sim[2]

        print "***Finish intrinsic evaluation. "


    def __init__(self, config, odm):
        self.config = config
        self.odm = odm
        self.build()




def sanity_SkipGramODM():
    odm = SkipGramODM(1)
    trainInput, trainLabel = odm.getTrainData()
    validInput, validLabel = odm.getValidData()
    testInput, testLabel = odm.getTestData()
    print len(trainInput), len(trainLabel)
    print len(validInput), len(validLabel)
    print len(testInput), len(testLabel)


def sanity_SkipGram():
    #lr, dim, h, WINDOW_SIZE, n_epochs,batch_size, fileToSaveWordVectors, dirToSaveModel, dirToLog
    config = Config(0.1, 30, 1, 10, 500, "./data/word2Vec" , "./saved_tf_model/", "./log_for_tensor_board" )
    odm = SkipGramODM(WINDOW_SIZE = config.WINDOW_SIZE)
    with tf.Graph().as_default():
        model = SkipGram(config, odm)
        with tf.Session() as session:
            # writer = tf.summary.FileWriter(config.dirToLog, session.graph)
            session.run( tf.global_variables_initializer() )
            model.fit(session)
            model.intrinsicEvaluation(session)
            # writer.close()


def tune():
    print "Parameters tuning will be done after speed problem is solved!"
    #TODO



if __name__ == "__main__":
    # sanity_SkipGramODM()
    sanity_SkipGram()
    # tune()
