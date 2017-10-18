import tensorflow as tf
import sys

class NLPM(object):
    def addPlaceHolder(self):
        with tf.variable_scope("hiddenLayer"):
            self.x = tf.placeholder(tf.int32, shape = [self.config.WINDOW_SIZE - 1])
        with tf.variable_scope("outputLayer"):
            self.y = tf.placeholder(tf.float32, shape = [self.config.V])
    def addVariable(self):
        with tf.variable_scope("hiddenLayer"):
            self.H = tf.get_variable("H", [self.config.V, self.config.dim])
            self.d = tf.get_variable("d", [self.config.m])
        with tf.variable_scope("outputLayer"):
            self.U = tf.get_variable("U", [self.config.m, self.config.V])
            self.b = tf.get_variable("b", [self.config.V])

    def getLossFunc(self):
        h = tf.tanh(tf.nn.embedding_lookup(self.H, self.x) + self.d)
        yHat = tf.matmul(h, self.U) + self.b
        yHat = tf.softmax(yHat)
        yHat = tf.reduce_prod(yHat, axis = 0)
        return  tf.reduce_sum( -tf.log( tf.slice(yHat, [self.y, self.y + 1]) ) )

    def getTrainOp(self, loss):
        return tf.train.GradientDescentOptimizer(self.config.lr).minimize(loss)

    def build():
        self.addPlaceHolder()
        self.addVariable()
        self.lossFunc = self.getLossFunc()
        self.trainOp = self.getTrainOp()
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
                print "Update!"
                fout = open(self.config.filePathWFV, "w")
                fout.write(self.C)
                fout.close()
    def __init__(self, config):
        self.config = config
        self.build()
