import tensorflow as tf
import sys

class NPLM(object):
    def addPlaceHolder(self):
        with tf.variable_scope("inputEmbeddingLayer"):
            self.cw = tf.placeholder(tf.int32, shape = [None, self.config.WINDOW_SIZE - 1])
        with tf.variable_scope("outputLayer"):
            self.pw = tf.placeholder(tf.int32, shape =  [None] )
    def addVariable(self):
        with tf.variable_scope("inputEmbeddingLayer"):
            self.C = tf.get_variable("C", [self.config.V, self.config.dim])
        with tf.variable_scope("hiddenLayer"):
            self.H = tf.get_variable("H", [self.config.dim * (self.config.WINDOW_SIZE - 1), self.config.h])
            self.d = tf.get_variable("d", [self.config.h])
        with tf.variable_scope("outputLayer"):
            self.U = tf.get_variable("U", [self.config.h, self.config.V])
            self.b = tf.get_variable("b", [self.config.V])

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
    def valid(self, sess, inputData, label):
        loss = sess.run(self.lossFunc,  {self.cw : inputData, self.pw : label})
        return loss
    def fit(self, sess):
        #writer = tf.summary.FileWriter("log", sess.graph)
        bestLoss = sys.float_info.max
        for epoch in xrange(self.config.n_epoches):
            print "***********Train epoch {:}****************".format(epoch)
            trainInput, trainLabel = self.config.dataTOLearnWFV.getTrain()
            trainLoss = self.train(sess, trainInput, trainLabel)
            validInput, validLabel = self.config.dataTOLearnWFV.getValid()
            validLoss = self.valid(sess, validInput, validLabel)
            print "***********epoch {:}, Train Loss {:}, Valid Loss {:}**************** ".format(epoch, trainLoss, validLoss)
            if validLoss < bestLoss:
                print "Better word feature vectors found! Writting to disk...."
                #TODO: write to disk instead of printing out
                print sess.run(self.C)
                # fout = open(self.config.filePathWFV, "w")
                # fout.write(self.C)
                # fout.close()
                print "File writting done!"
        #writer.close()
    def __init__(self, config):
        self.config = config
        self.build()
