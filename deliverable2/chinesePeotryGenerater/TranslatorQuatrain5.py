import tensorflow as tf
import Model

class Config(object):
    q = 5
    v = 30
    lr = 0.5

class TranslatorQuatrain5(Model):
    def addPlaceHolder(self):
        self.line0 = tf.placeholder( tf.int32, shape = [5,] )
        self.line123 = tf.placeholder( tf.int32, shape = [5,3] )
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
        To get the probability distribution of each char in line2, 3 and 4.
        Those probability distributions are used to get cross entropy loss.

        CSM(convolutional sentence model) converts a line to a vector.

        RCM(recurrent context model) converts line vectors to context vector.

        RGM(recurrent generation model) uses context vector from previous lines
        and chars from the current line to generate the current vector.
        Once a new line is generated from RGM, it goes into CSM again.......
        """
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
        h = tf.sigmoid(tf.mul(self.M, tf.concat([v1, h], 0)))# vector of size q
        cv1 = rcm(h)
        # RGM: generate line2
        dist2, line2 = rgm(cv1)
        # CSM: line2 to vector
        v2 = csm(line2)
        # RCM: context of line0 + line1 + line2
        h = tf.sigmoid(tf.mul(self.M, tf.concat([v2, h], 0)))# vector of size q
        cv2 = rcm(h)
        # RGM: generate line3
        dist3, _ = rgm(cv2)

        # return predicted function
        return tf.concat([dist1, dist2, dist3], axis = -1) # 5 * 3 * v

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
        cv: q * 4. Row i represents the context for generating the i + 1 th char

        output:
        dist:
        line:
        """
        r = tf.zeros([self.config.q])
        y = tf.TensorArray()
        w = tf.TensorArray()
        for i in xrange(4):
            y[i] = tf.mul(self.Y, r)
            w[i] = tf.argmax(y0)
            r = tf.sigmoid( tf.mul(self.R, r) + tf.nn.embedding_lookup(self.X, [w[i]]) + tf.mul(self.X, tf.nn.embedding_lookup(cv, [i])) )
        dist = tf.concat(y, axis = 1)
        line = tf.concat(w, axis = 0)
        return dist, line

    def getLossFunc(self, predFunc):
        loss = tf.constant(0)
        for i in xrange(predFunc.shape[0]):
            for j in xrange(predFunc.shape[1]):
                loss -= tf.log(predFunc[i][j][self.line123[i][j]])
        return loss
    def getTrainOp(self, loss):
        optimizer = tf.train.AdamOptimizer(self.config.lr)
        train_op = optimizer.minimize(loss)
        return train_op

    def train(self, sess, inputTrain, labelTrain):
        sess.run(self.train_op, {self.line0 : inputTrain, self.line123 : labelTrain})

    def predict(self, sess, inputPred):
        labelPred = sess.run(self.predFunc, {self.line0 : inputPred
        return labelPred

    def __init__(self, config):
        self.config = Config()
        self.build()

def sanity_test():
    model = TranslatorQuatrain5()
    sess= tf.Session()
    inputTrain = tf.constant([1,2,3,4,5])
    labelTrain = tf.constant([ [1,2,3,4,5], [6,7,8,9,0], [2,4,6,8,0] ])
    inputPred = tf.constant([1,3,5,7,9])
    model.train(sess, inputTrain, labelTrain)
    labelPred = model.predict(sess, inputPred)
    print labelPred

if __name__ == "__main__":
    sanity_test()
