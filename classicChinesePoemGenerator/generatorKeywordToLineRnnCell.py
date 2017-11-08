import tensorflow as tf
import numpy as np

class RNNCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, input_size, state_size, output_size):
        self.input_size = input_size
        self._state_size = state_size
        self._output_size = output_size

        with tf.variable_scope(type(self).__name__):
            xavier_initializer = tf.contrib.layers.xavier_initializer()
            self.W_xh = tf.get_variable("W_xh", initializer = xavier_initializer, shape = (self.input_size, self.state_size), dtype = tf.float32)
            self.W_hh = tf.get_variable("W_hh", initializer = xavier_initializer, shape = (self.state_size, self.state_size), dtype = tf.float32)
            self.b_h = tf.get_variable("b_h", initializer = xavier_initializer, shape = (self.state_size,), dtype = tf.float32)
            self.W_hd = tf.get_variable("W_hd", initializer = xavier_initializer, shape = (self.state_size, self.output_size), dtype = tf.float32)
            self.b_d = tf.get_variable("b_d", initializer = xavier_initializer, shape = (self.output_size,), dtype = tf.float32)

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def __call__(self, inputs, state, scope=None):
        """
        Update hidden state one time.

        Formula:

        new_state = sigmoid( inputs * W_xh + state * W_hh + b_h)
        output = softmax( new_state * W_hd + b_d)

        Args:
            inputs: [None, input_size]
            state: [None, state_size]
        Returns:
            outputs : [None, output_size]
            new_state: [None, state_size]

        """
        new_state = tf.nn.sigmoid( tf.matmul(inputs, self.W_xh ) + tf.matmul(state, self.W_hh) + self.b_h)
        output = tf.nn.softmax( tf.matmul(new_state, self.W_hd) + self.b_d )
        return output, new_state

def sanity_RNNCell():
    with tf.Graph().as_default():
        with tf.variable_scope("sanity_RNNCell"):
            cell = RNNCell(3, 2, 4)
            inputs_placeholder = tf.placeholder(tf.float32, shape=(None,3))
            state_placeholder = tf.placeholder(tf.float32, shape=(None,2))
            inputs = np.array([
                [0.4, 0.5, 0.6],
                [0.3, -0.2, -0.1]], dtype=np.float32)
            state = np.array([
                [0.2, 0.5],
                [-0.3, -0.3]], dtype=np.float32)
            for i in xrange(2):
                output, new_state = cell(inputs_placeholder, state_placeholder, scope="rnn")
                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())
                    print sess.run((output, new_state), {inputs_placeholder : inputs, state_placeholder : state})


if __name__ == "__main__":
    sanity_RNNCell()
