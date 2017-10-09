import tensorflow as tf
import numpy as np

b = 6 / tf.constant([1,2])
c = [6] / tf.constant([1,2])
d = np.array([6]) / tf.constant([1,2])
e = tf.constant([6]) / tf.constant([1,2])
print b #Tensor("div:0", shape=(2,), dtype=int32)
print c #Tensor("div_1:0", shape=(2,), dtype=int32)
print d #Tensor("div_2:0", shape=(2,), dtype=int32)
print e #Tensor("div_3:0", shape=(2,), dtype=int32)


f = 6 / np.array([1,2])
g = [6] / np.array([1,2])
print type(f) #<type 'numpy.ndarray'>
print type(g) #<type 'numpy.ndarray'>


# sess = tf.Session()
# print sess.run(b) #[6 3]
