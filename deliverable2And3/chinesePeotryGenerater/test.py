import tensorflow as tf
import sys

x = tf.nn.embedding_lookup([1,2,3], 2)
sess = tf.Session()

print x
print sess.run(x)
