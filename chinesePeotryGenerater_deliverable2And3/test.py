import tensorflow as tf
import sys
sess = tf.Session()


logits = tf.constant([[0.0,0], [1,1], [2,2], [3,3], [4,4]])

x = tf.shape(logits)
# print x[0]
# print sess.run(x[0])

y = tf.zeros(tf.reshape(x[0], [-1]))
# z = tf.zeros([x[0]])
print y
print sess.run(y)
# print z
