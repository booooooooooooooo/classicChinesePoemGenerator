import tensorflow as tf

sess = tf.Session()

# T1 = tf.constant([[1,2,3], [4,5,6], [7,8,9]])
# C12 = tf.constant([[2,2,2], [2,2,2]])
#
# print tf.reduce_sum(tf.nn.embedding_lookup(T1, [0, 1]) * C12, axis = 0)
# print sess.run( tf.reduce_sum(tf.nn.embedding_lookup(T1, [0, 1]) * C12, axis = 0) )
#
# print [tf.reduce_sum(tf.nn.embedding_lookup(T1, [i, i+1]) * C12, axis = 0) for i in xrange(2)]
# print sess.run( [tf.reduce_sum(tf.nn.embedding_lookup(T1, [i, i+1]) * C12, axis = 0) for i in xrange(2)] )
#
# print tf.concat([tf.reshape(tf.reduce_sum(tf.nn.embedding_lookup(T1, [i, i+1]) * C12, axis = 0), [1,3]) for i in xrange(2)], axis = 0)
# print sess.run( tf.concat([tf.reshape(tf.reduce_sum(tf.nn.embedding_lookup(T1, [i, i+1]) * C12, axis = 0), [1,3]) for i in xrange(2)], axis = 0))



# a = [[1,2,3], [4,5,6]]
# b = [[1,1], [2,2], [3,3]]
# print sess.run(tf.matmul(a, b))
# print sess.run(tf.multiply(a, a))



a = [[1,2,3], [4,5,6]]
b = [[2], [2], [2]]
print tf.matmul(a, b)
