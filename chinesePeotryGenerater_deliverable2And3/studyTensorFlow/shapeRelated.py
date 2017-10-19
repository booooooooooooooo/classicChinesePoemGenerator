import tensorflow as tf
import sys
sess = tf.Session()

#Problem: How to use a None dimension to reshape another tensor
cw = tf.placeholder(tf.int32, shape = [None, 5])
pw = tf.placeholder(tf.int32, shape = tf.reshape( tf.shape(cw)[0] , [-1]) )
