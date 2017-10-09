import tensorflow as tf
import Model

class Config(object):
    q = 5
    v = 30

class TranslatorQuatrain5(Model):
    def addPlaceHolder(self):

    def addVariable(self):

    def getPredFunc(self):

    def getLossFunc(self, predFunc):

    def getTrainOp(self, loss):

    def train(self, sess, inputTrain, labelTrain):

    def predict(self, sess, inputPred):

    def __init__(self, config):
        self.config = Config()
        self.build()

def tuneTranslatorQuatrain5():
    model = TranslatorQuatrain5()
    sess= tf.Session()
    #TODO: make dummy inputTrain, labelTrain and inputPred
    model.train(sess, inputTrain, labelTrain)
    pred = model.predict(sess, inputPred)
    print pred
