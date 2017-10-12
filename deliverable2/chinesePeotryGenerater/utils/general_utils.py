import tensorflow as tf

def getOneHot(index, v):
    a = np.zeros([v])
    a[index] = 1
    return tf.constant(a)
