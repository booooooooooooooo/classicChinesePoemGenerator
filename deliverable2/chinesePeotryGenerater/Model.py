class Model(object):
    def addPlaceHolder(self):
        
    def addVariable(self):

    def getPredFunc(self):

    def getLossFunc(self, predFunc):

    def getTrainOp(self, loss):

    def build():
        self.addPlaceHolder()
        self.addVariable()
        self.predFunc = self.getPredFunc()
        self.lossFunc = self.getLossFunc(self.predFunc)
        self.trainOp = self.getTrainOp()

    def train(self, sess, input, label):

    def predict(self, sess, input):
