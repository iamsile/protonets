import typing
import tensorflow as tf
import numpy as np


def conv_block(inputs, out_channels, name='conv'):
    with tf.variable_scope(name):
        conv = tf.layers.conv2d(inputs, out_channels,
                                kernel_size=3, padding='SAME')
        conv = tf.contrib.layers.batch_norm(
            conv, updates_collections=None, decay=0.99, scale=True, center=True)
        conv = tf.nn.relu(conv)
        conv = tf.contrib.layers.max_pool2d(conv, 2)
        return conv


def encoder(x, h_dim, z_dim, reuse=False):
    with tf.variable_scope('encoder', reuse=reuse):
        net = conv_block(x, h_dim, name='conv_1')
        net = conv_block(net, h_dim, name='conv_2')
        net = conv_block(net, h_dim, name='conv_3')
        net = conv_block(net, z_dim, name='conv_4')
        net = tf.contrib.layers.flatten(net)
        return net


def euclidean_distance(a, b):
    # a.shape = N x D
    # b.shape = M x D
    N, D = tf.shape(a)[0], tf.shape(a)[1]
    M = tf.shape(b)[0]
    a = tf.tile(tf.expand_dims(a, axis=1), (1, M, 1))
    b = tf.tile(tf.expand_dims(b, axis=0), (N, 1, 1))
    return tf.reduce_mean(tf.square(a - b), axis=2)


class Protonet(object):

    def __init__(self, image_dimensions: typing.Iterable[int],
                 channels: int, h_dim: int = 64, z_dim: int = 64):
        self.x = tf.placeholder(
            tf.float32, [None, None, *image_dimensions, channels])
        self.q = tf.placeholder(
            tf.float32, [None, None, *image_dimensions, channels])
        x_shape = tf.shape(self.x)
        num_classes, num_support = x_shape[0], x_shape[1]
        num_query = tf.shape(self.q)[1]
        self.y = tf.placeholder(tf.int64, [None, None])
        y_one_hot = tf.one_hot(self.y, depth=num_classes)
        emb_x = encoder(tf.reshape(
            self.x, [num_classes * num_support, *image_dimensions, channels]), h_dim, z_dim)
        emb_dim = tf.shape(emb_x)[-1]
        emb_x = tf.reduce_mean(tf.reshape(
            emb_x, [num_classes, num_support, emb_dim]), axis=1)
        emb_q = encoder(tf.reshape(
            self.q, [num_classes * num_query, *image_dimensions, channels]), h_dim, z_dim, reuse=True)
        dists = euclidean_distance(emb_q, emb_x)
        log_p_y = tf.reshape(tf.nn.log_softmax(-dists),
                             [num_classes, num_query, -1])
        self.loss = - \
            tf.reduce_mean(tf.reshape(tf.reduce_sum(
                tf.multiply(y_one_hot, log_p_y), axis=-1), [-1]))
        self.acc = tf.reduce_mean(tf.to_float(
            tf.equal(tf.argmax(log_p_y, axis=-1), self.y)))
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def train(self, support: np.ndarray, query: np.ndarray, labels: np.ndarray):
        """Performs a one optimization step and returns loss & accuracy."""
        _, loss, accuracy = self.sess.run([self.train_op, self.loss, self.acc], feed_dict={
            self.x: support, self.q: query, self.y: labels})
        return loss, accuracy

    def evaluate(self, support: np.ndarray, query: np.ndarray, labels: np.ndarray):
        """Evaluates performance of model on data."""
        loss, accuracy = self.sess.run([self.loss, self.acc], feed_dict={
            self.x: support, self.q: query, self.y: labels})
        return loss, accuracy
