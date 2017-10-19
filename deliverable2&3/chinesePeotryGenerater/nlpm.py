import tensorflow as tf
import sys

class NLPM(object):
    def addPlaceHolder(self):
        with tf.variable_scope("inputEmbeddingLayer"):
            self.cw = tf.placeholder(tf.int32, shape = [self.config.WINDOW_SIZE - 1])
        with tf.variable_scope("outputLayer"):
            self.pw = tf.placeholder(tf.float32, shape = [])
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
        x = tf.reshape( tf.nn.embedding_lookup(self.C, self.cw), [-1] ) # (dim*(WINDOW_SIZE - 1) ) *
        z1 = self.d + tf.matmul(self.x, self.H)
        h1 = tf.tanh(z1)
        z2 = self.b + tf.matmul(h1, self.U)
        yHat = tf.softmax(z2)
        return  tf.reduce_sum( -tf.log( tf.slice(yHat, [self.pw], [1] ) ) )

    def getTrainOp(self, loss):
        return tf.train.GradientDescentOptimizer(self.config.lr).minimize(loss)

    def build():
        self.addPlaceHolder()
        self.addVariable()
        self.lossFunc = self.getLossFunc()
        self.trainOp = self.getTrainOp(self.lossFunc)

    def train(self, sess, input, label):
        _ , loss = sess.run([self.trainOp, self.lossFunc], {self.x : input, self.y : label})
        return loss
    def valid(self, sess, input, label):
        loss = sess.run(self.lossFunc,  {self.x : input, self.y : label})
        return loss
    def fit(self, sess):
        bestLoss = sys.float_info.max
        for epoch in self.config.n_epoches:
            trainInput, trainLabel = self.config.dataTOLearnWFV.getTrain()
            trainLoss = self.train(sess, trainInput, trainLabel)
            validInput, validLabel = self.config.dataTOLearnWFV.getValid()
            validLoss = self.valid(sess, validInput, validLabel)
            print "epoch " + epoch + "Train Loss " + trainLoss + "ValidLoss " + validLoss
            if validLoss < bestLoss:
                print "Better word feature vectors found! Writting to disk...."
                fout = open(self.config.filePathWFV, "w")
                fout.write(self.C)
                fout.close()
                print "File writting done!"
    def __init__(self, config):
        self.config = config
        self.build()
