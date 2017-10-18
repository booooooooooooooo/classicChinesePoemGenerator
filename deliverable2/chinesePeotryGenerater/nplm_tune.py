

class Config(object):
    def __init__(self, V, m, n, h, lr, n_epoches, filePathWFV, dataTOLearnWFV):
        self.V = V
        self.m = m
        self.n = n
        self.h = h
        self.lr = lr
        self.n_epoches = n_epoches
        self.filePathWFV = filePathWFV
        self.dataTOLearnWFV = dataTOLearnWFV


def tune():
    dataTOLearnWFV = DataToLearnWFV()
    #TODO: how to handle tf when iterating over diffrent configs
    config = Config()#TODO: give parameters to Config
    with tf.Graph().as_default():
        model = NLPM(config)
        with tf.Session() as session:
            init = tf.global_variables_initializer()
            session.run(init)
            model.fit(session)
