import tensorflow as tf

a = tf.constant([1,2,3])
sess = tf.Session()
print tf.argmax(a)
print sess.run( tf.argmax(a) )
