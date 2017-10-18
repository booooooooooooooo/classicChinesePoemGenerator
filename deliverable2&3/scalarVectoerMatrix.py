import tensorflow as tf


sess = tf.Session()

# scalar has shape ()
a1 = tf.constant(5) # scalar
a2 = tf.constant([5]) #shape=(1,)
a3 = tf.constant([5], shape = (1,1))#shape=(1, 1)
print a1
print a2
print a3


# scalar has shape = None
b1 = tf.placeholder(tf.float32) # scalar
b2 = tf.placeholder(tf.float32, shape = 3) # shape=(3,)
b3 = tf.placeholder(tf.float32, shape = (3))# shape=(3,)
b4 = tf.placeholder(tf.float32, shape = (1,3))# shape=(1, 3)
print b1
print b2
print b3
print b4


# broadcasting chooses the more complex shape
h = [1,2,3]
print sess.run(tf.constant(2) * h)
print sess.run(tf.constant([2]) * h)
print sess.run(tf.constant([[2,2,2]]) * h)
print sess.run(tf.constant([[2,2,2],[2,2,2]]) * h)
