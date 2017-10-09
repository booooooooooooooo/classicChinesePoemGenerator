import tensorflow as tf
import numpy as np

sample_size = 80
feature_size = 10
class_size = 20
lr = 0.5

## Construct a model with only one softmax activation layer.
x = tf.placeholder(tf.float32, (sample_size, feature_size))
y = tf.placeholder(tf.float32, (sample_size, class_size))

w = tf.Variable(tf.zeros((feature_size, class_size)))
b = tf.Variable(tf.zeros((class_size,)))

z = tf.add( tf.matmul(x, w), b )
y_hat = tf.nn.softmax(z)

## Define loss function
loss_ce = -tf.reduce_sum( y * tf.log(y_hat) ) / sample_size

## Define optimization.
optimizer = tf.train.GradientDescentOptimizer(lr)
train_op = optimizer.minimize(loss_ce)

## Define train set
inputs = np.random.rand(sample_size, feature_size)
ground_truth = np.zeros((sample_size, class_size), dtype=np.int32)
ground_truth[:, 0] = 1

feed = {x : inputs, y : ground_truth }
## Run
sess = tf.Session() #stateful
sess.run(tf.global_variables_initializer() )
for i in range(1000):
    _, loss = sess.run([train_op, loss_ce], feed)
    print loss
