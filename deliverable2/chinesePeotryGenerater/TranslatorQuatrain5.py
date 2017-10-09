import tensorflow as tf
import Model

class Config(object):
    q = 5
    v = 30
    lr = 0.5

class TranslatorQuatrain5(Model):
    def addPlaceHolder(self):
        self.line0 = tf.placeholder( tf.int32, shape = (5,) )
        self.line123 = tf.placeholder( tf.int32, shape = (5,3) )
    def addVariable(self):
        #CSM
        self.L = tf.Variable( tf.random_normal( shape = (self.config.v, self.config.q) )# TODO: normalize colume? BUT cannot keep length = 1 when trainning
        self.C12 = tf.Variable( tf.random_normal( shape = (2, self.config.q ) ) )
        self.C22 = tf.Variable( tf.random_normal( shape = (2, self.config.q ) ) )
        self.C33 = tf.Variable( tf.random_normal( shape = (3, self.config.q ) ) )
        #RCM
        self.M = tf.Variable( tf.random_normal( shape = (self.config.q, self.config.q * 2) ) )
        self.U0 = tf.Variable( tf.random_normal( shape = (self.config.q, self.config.q) ) )
        self.U1 = tf.Variable( tf.random_normal( shape = (self.config.q, self.config.q) ) )
        self.U2 = tf.Variable( tf.random_normal( shape = (self.config.q, self.config.q) ) )
        self.U3 = tf.Variable( tf.random_normal( shape = (self.config.q, self.config.q) ) )
        #RGM
        self.R = tf.Variable( tf.random_normal( shape = (self.config.q, self.config.q) ) )
        self.V = tf.Variable( tf.random_normal( shape = (self.config.q, self.config.v) ) )
        self.H = tf.Variable( tf.random_normal( shape = (self.config.q, self.config.q) ) )
        self.Y = tf.Variable( tf.random_normal( shape = (self.config.v, self.config.q) ) )
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
        v0 = csm(self.line0) # vector of size q
        # RCM: context of line0
        h = tf.zeros((self.config.q))# vector of size q
        h = tf.sigmoid(tf.mul(self.M, tf.concat([v0, h], 0)))# vector of size q
        cv0 = rcm(h)
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
        T1 = tf.nn.embedding_lookup(self.L, lineX) # 5 * q
        T2 = tf.concat([tf.reduce_sum(tf.nn.embedding_lookup(T1, [i, i + 1]) * self.C12, axis = 0) for i in xrange(4)], axis = 1)# 4 * q
        T3 = tf.concat([tf.reduce_sum(tf.nn.embedding_lookup(T2, [i, i + 1]) * self.C22, axis = 0) for i in xrange(3)], axis = 1)# 3 * q
        T4 = tf.reduce_sum( tf.nn.embedding_lookup(T3, [0, 2]) ) * self.C23, axis = 0) , axis = 1)# vector of size q
    def rcm(hs):
        u00 = tf.sigmoid(tf.mul(self.U0, hs)) # q*q
        u01 = tf.sigmoid(tf.mul(self.U1, hs))# q*q
        u02 = tf.sigmoid(tf.mul(self.U2, hs))# q*q
        u03 = tf.sigmoid(tf.mul(self.U3, hs))# q*q
        return tf.concat([u00, u01, u02, u03], axis = 1)
    def rgm(cv):
        r = tf.zeros((self.config.q,))
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
