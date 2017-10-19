import tensorflow as tf
from utils.dataToLearnWFV import DataToLearnWFV
from nplm import NPLM


class Config(object):
    def __init__(self, V, dim, WINDOW_SIZE, h, lr, n_epoches, filePathWFV, dataTOLearnWFV):
        self.V = V
        self.dim = dim
        self.WINDOW_SIZE = WINDOW_SIZE
        self.h = h
        self.lr = lr
        self.n_epoches = n_epoches
        self.filePathWFV = filePathWFV
        self.dataTOLearnWFV = dataTOLearnWFV


# def tune():
#     dataTOLearnWFV = DataToLearnWFV()
#     #TODO: how to handle tf when iterating over diffrent configs
#     config = Config()#TODO: give parameters to Config
#     with tf.Graph().as_default():
#         model = NPLM(config)
#         with tf.Session() as session:
#             init = tf.global_variables_initializer()
#             session.run(init)
#             model.fit(session)

def sanity_check():
    dataTOLearnWFV = DataToLearnWFV()
    config = Config(dataTOLearnWFV.getV(), 5, dataTOLearnWFV.WINDOW_SIZE, 10, 0.5, 3, "./utils/data/boWFV/wfv", dataTOLearnWFV )#TODO: give parameters to Config
    with tf.Graph().as_default():
        model = NPLM(config)
        with tf.Session() as session:
            init = tf.global_variables_initializer()
            session.run(init)
            model.fit(session)

if __name__ == "__main__":
    sanity_check()
