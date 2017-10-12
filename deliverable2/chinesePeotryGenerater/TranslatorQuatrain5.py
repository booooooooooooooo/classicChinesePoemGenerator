import tensorflow as tf
import Model
from utils.general_utils import *

class Config(object):
    q = 5
    v = 30
    lr = 0.5

class TranslatorQuatrain5(Model):
    def addPlaceHolder(self):
        self.line0 = tf.placeholder( tf.int32, shape = [5] )
        self.line123 = tf.placeholder( tf.int32, shape = [15] )
    def addVariable(self):
        q = self.config.q
        v = self.config.v
        lr = self.config.lr
        #CSM
        with tf.variable_scope("csm"):
            self.L = tf.get_variable("L", [v,q])
            self.C12 = tf.get_variable("C12", [2, q])
            self.C22 = tf.get_variable("C22", [2, q])
            self.C33 = tf.get_variable("C33", [3, q])
        #RCM
        with tf.variable_scope("rcm"):
            self.M = tf.get_variable("M", [q, q * 2])
            self.U0 = tf.get_variable("U0", [q, q])
            self.U1 = tf.get_variable("U1", [q, q])
            self.U2 = tf.get_variable("U2", [q, q])
            self.U3 = tf.get_variable("U3", [q, q])
        #RGM
        with tf.variable_scope("rgm"):
            self.R = tf.get_variable("R", [q, q])
            self.V = tf.get_variable("V", [q, v])
            self.H = tf.get_variable("H", [q, q])
            self.Y = tf.get_variable("Y", [v, q])
    def getPredFunc(self):
        """
        To get the probability distribution of line234. The line is also returned for convenience.

        Those probability distributions are used to get cross entropy loss.

        CSM(convolutional sentence model) converts a line to a vector.

        RCM(recurrent context model) converts line vectors to context vector.

        RGM(recurrent generation model) uses context vector from previous lines
        and chars from the current line to generate the current vector.
        Once a new line is generated from RGM, it goes into CSM again.......
        """
        #TODO: do this function in TF graph, e.g. use TensorArray instead of []
        #TODO: use a iteration to beautify this part
        # CSM: line0 to vector
        v0 = csm(self.line0) # q *
        # RCM: context of line0
        h = tf.zeros((self.config.q))# q *
        h = tf.sigmoid(tf.matmul(self.M, tf.concat([v0, h], axis = 0)))# q *
        cv0 = rcm(h) # q * 4
        # RGM: generate line1
        dist1, line1  = rgm(cv0)
        # CSM: line1 to vector
        v1 = csm(line1)
        # RCM: context of line0 + line1
        h = tf.sigmoid(tf.matmul(self.M, tf.concat([v1, h], axis = 0)))#q*
        cv1 = rcm(h)
        # RGM: generate line2
        dist2, line2 = rgm(cv1)
        # CSM: line2 to vector
        v2 = csm(line2)
        # RCM: context of line0 + line1 + line2
        h = tf.sigmoid(tf.matmul(self.M, tf.concat([v2, h], axis = 0)))#q*
        cv2 = rcm(h)
        # RGM: generate line3
        dist3, line3 = rgm(cv2)

        # return predicted function
        return tf.concat([dist1, dist2, dist3], axis = 0), tf.concat([line1, line2, line3], axis = 0) # 15*v, 15*

    def csm(lineX):
        """
        Use a convolutional network to convert a line of 5 char vector to a single vector

        The filter sizes are q * 2 for C12, q * 2 for C22, and q * 3 for C33.
        """
        T1 = tf.nn.embedding_lookup(self.L, lineX) # 5 * q
        arrT2 = [tf.nn.sigmoid( tf.reshape(tf.reduce_sum(tf.nn.embedding_lookup(T1, [i, i + 1]) * self.C12, axis = 0) , [1, q]) ) for i in xrange(4) ]
        T2 = tf.concat(arrT2, axis = 0)# 4 * q
        arrT3 = [tf.nn.sigmoid( tf.reshape(tf.reduce_sum(tf.nn.embedding_lookup(T2, [i, i + 1]) * self.C22, axis = 0) , [1, q]) ) for i in xrange(3) ]
        T3 = tf.concat(arrT3, axis = 0)# 3 * q
        T4 = tf.nn.sigmoid( tf.reduce_sum( T3 * self.C33, axis = 0) ) # q *
        return T4

    def rcm(hs):
        """
        ui is the context used to generate the (i+1)th char in the next line.
        """
        u0 = tf.sigmoid(tf.matmul(self.U0, hs)) # q * 1
        u1 = tf.sigmoid(tf.matmul(self.U1, hs))# q * 1
        u2 = tf.sigmoid(tf.matmul(self.U2, hs))# q * 1
        u3 = tf.sigmoid(tf.matmul(self.U3, hs))# q * 1
        return tf.concat([u0, u1, u2, u3], axis = 1)
    def rgm(cv):
        """
        Get the distribution of each position. Also get the predicted char for convenience.

        input
        cv: a Tensor with shape q * 4. Row i represents the context for generating the i + 1 th char

        output:

        dist: a Tensor with shape 5 * v. Row i represents the prob distribution of the ith char.
        line: a Tensor with shape 5 *. Entry i represents the ith char's index in V
        """
        prob = tf.constant(0.0)
        dist = tf.constant([5, self.config.v])
        line = tf.constant([5])
        for i in xrange(self.config.v):
            prob_, dist_, line_ = rgmHelper(i, cv)
            if prob_ > prob:
                prob = prob_
                dist = dist_
                line = line_
        return dist, line

    def rgmHelper(i, cv):
        #TODO: do this function in TF graph, e.g. use TensorArray instead of []
        y = [getOneHot(i, self.config.v)]
        w = [i]

        r = tf.zeros([self.config.q])
        for i in xrange(4):
            r = tf.sigmoid( tf.matmul(self.R, r) + tf.nn.embedding_lookup(self.X, [w[i]]) + tf.matmul(self.H, tf.nn.embedding_lookup(cv, [i])) )
            y.append( tf.matmul(self.Y, r) )
            w.append( tf.argmax( y[i + 1] ) )
        line = tf.constant(w)#5*
        dist = tf.concat([tf.reshape(tf.nn.softmax(x), [1, self.config.v]) for x in y], axis = 0)#5*v
        prob = 1.0
        for i in xrange(5):
            prob *= dist[i][line[i]]
        return prob, dist, line


    def getLossFunc(self, predFunc):
        dist, _ = predFunc # dist: 15 * v
        loss = tf.reduce_mean([ -tf.log(dist[i, self.line234[i]]) for i in xrange(15)])
        return loss

    def getTrainOp(self, loss):
        optimizer = tf.train.AdamOptimizer(self.config.lr)
        train_op = optimizer.minimize(loss)
        return train_op

    def train(self, sess, inputTrain, labelTrain):
        sess.run(self.train_op, {self.line0 : inputTrain, self.line123 : labelTrain})

    def predict(self, sess, inputPred):
        dist, line = sess.run(self.predFunc, {self.line0 : inputPred
        return line

    def __init__(self, config):
        self.config = Config()
        self.build()

def sanity_test():
    model = TranslatorQuatrain5()
    sess= tf.Session()
    inputTrain = tf.constant([1,2,3,4,5])
    labelTrain = tf.constant([ 1,2,3,4,5,6,7,8,9,0,2,4,6,8,0 ])
    inputPred = tf.constant([1,3,5,7,9])
    for i in xrange(1000):
        model.train(sess, inputTrain, labelTrain)
    labelPred = model.predict(sess, inputPred)

    print labelPred

if __name__ == "__main__":
    sanity_test()
