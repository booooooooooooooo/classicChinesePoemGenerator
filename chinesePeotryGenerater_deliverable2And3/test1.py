import tensorflow as tf
import sys
import pickle


tf.reset_default_graph()

# Create some variables.
v1 = tf.get_variable("v1", shape=[3])
# v2 = tf.get_variable("v2", shape=[5])

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
  # Restore variables from disk.
  saver.restore(sess, "./utils/data/boWFV/wfv")
  print("Model restored.")
  # Check the values of the variables
  print("v1 : %s" % sess.run(v1))
  # print("v2 : %s" % v2.eval())
