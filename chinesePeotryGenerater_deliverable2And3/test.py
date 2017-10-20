import tensorflow as tf
import sys
import pickle



with tf.variable_scope("embeddingLayer"):
    a1 = tf.get_variable("a1", [3])
with tf.variable_scope("embeddingLayer"):
    a2 = tf.get_variable("a2", [3])
a3 = a1 + a2

sess = tf.Session()

writer = tf.summary.FileWriter("log", sess.graph)
sess.run(tf.global_variables_initializer())
sess.run(a3)
writer.close()
