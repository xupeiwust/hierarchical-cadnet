"""Module containing classes for different neural network layers for Hierarchical CADNet neural architecture."""


import tensorflow as tf
import math


class GraphCNNGlobal(object):
    BN_DECAY = 0.999
    GRAPHCNN_INIT_FACTOR = 1.
    GRAPHCNN_I_FACTOR = 1.0
    

class GraphCNNLayer(tf.keras.layers.Layer):
    """Graph convolutional layer that uses the adjacency matrix."""
    def __init__(self, filters, name="GCNN", **kwargs):
        super(GraphCNNLayer, self).__init__(name=name, **kwargs)

        self.num_filters = filters
        self.weight_decay = 0.0005
        self.W = None
        self.W_I = None
        self.b = None

    def build(self, input_shape):
        # num_features = (c)
        # num_nodes = (n)
        # num_filters = (j)
        # W_dim = (c x j)
        # W_I_dim = (c x j)
        # b_dim = (n x j)

        V_shape, _ = input_shape
        num_features = V_shape[1]
        W_dim = [num_features, self.num_filters]
        W_I_dim = [num_features, self.num_filters]
        b_dim = [self.num_filters]

        W_stddev = math.sqrt(1.0 / num_features * GraphCNNGlobal.GRAPHCNN_INIT_FACTOR)
        W_I_stddev = math.sqrt(
            GraphCNNGlobal.GRAPHCNN_I_FACTOR / num_features * GraphCNNGlobal.GRAPHCNN_INIT_FACTOR)

        self.W = self.add_weight(
            shape=W_dim,
            dtype=tf.float32,
            initializer=tf.keras.initializers.TruncatedNormal(stddev=W_stddev),
            regularizer=tf.keras.regularizers.l2(self.weight_decay),
            trainable=True,
            name="W")

        self.W_I = self.add_weight(
            shape=W_I_dim,
            dtype=tf.float32,
            initializer=tf.keras.initializers.TruncatedNormal(stddev=W_I_stddev),
            regularizer=tf.keras.regularizers.l2(self.weight_decay),
            trainable=True,
            name="W_I")

        self.b = self.add_weight(
            shape=b_dim,
            dtype=tf.float32,
            initializer=tf.keras.initializers.Constant(0.1),
            trainable=True,
            name="bias")

    def call(self, input, training=None):
        V, A = input
        n = tf.matmul(A, V)
        output = tf.matmul(n, self.W) + tf.matmul(V, self.W_I) + self.b

        return output


class GraphEdgeConvLayer(tf.keras.layers.Layer):
    """Graph convolutional layer that uses the edge convexity."""
    def __init__(self, filters, name="GCNN_Edge", **kwargs):
        super(GraphEdgeConvLayer, self).__init__(name=name, **kwargs)

        self.num_filters = filters
        self.weight_decay = 0.0005
        self.W_E_1 = None
        self.W_E_2 = None
        self.W_E_3 = None
        self.W_I = None
        self.b = None

    def build(self, input_shape):
        V_shape, _, _, _ = input_shape
        num_features = V_shape[1]
        W_dim = [num_features, self.num_filters]
        b_dim = [self.num_filters]

        W_stddev = math.sqrt(1.0 / num_features * GraphCNNGlobal.GRAPHCNN_INIT_FACTOR)
        W_I_stddev = math.sqrt(
            GraphCNNGlobal.GRAPHCNN_I_FACTOR / num_features * GraphCNNGlobal.GRAPHCNN_INIT_FACTOR)

        self.W_E_1 = self.add_weight(
            shape=W_dim,
            dtype=tf.float32,
            initializer=tf.keras.initializers.TruncatedNormal(stddev=W_stddev),
            regularizer=tf.keras.regularizers.l2(self.weight_decay),
            trainable=True,
            name="W_E_1")

        self.W_E_2 = self.add_weight(
            shape=W_dim,
            dtype=tf.float32,
            initializer=tf.keras.initializers.TruncatedNormal(stddev=W_stddev),
            regularizer=tf.keras.regularizers.l2(self.weight_decay),
            trainable=True,
            name="W_E_2")

        self.W_E_3 = self.add_weight(
            shape=W_dim,
            dtype=tf.float32,
            initializer=tf.keras.initializers.TruncatedNormal(stddev=W_stddev),
            regularizer=tf.keras.regularizers.l2(self.weight_decay),
            trainable=True,
            name="W_E_3")

        self.W_I = self.add_weight(
            shape=W_dim,
            dtype=tf.float32,
            initializer=tf.keras.initializers.TruncatedNormal(stddev=W_I_stddev),
            regularizer=tf.keras.regularizers.l2(self.weight_decay),
            trainable=True,
            name="W_I")

        self.b = self.add_weight(
            shape=b_dim,
            dtype=tf.float32,
            initializer=tf.keras.initializers.Constant(0.1),
            trainable=True,
            name="bias")

    def call(self, input, training=None):
        V, E_1, E_2, E_3 = input
        n_E_1 = tf.matmul(E_1, V)
        n_E_2 = tf.matmul(E_2, V)
        n_E_3 = tf.matmul(E_3, V)

        output = tf.matmul(n_E_1, self.W_E_1) + tf.matmul(n_E_2, self.W_E_2) + tf.matmul(n_E_3, self.W_E_3) + \
                 tf.matmul(V, self.W_I) + self.b
        return output


class GraphEmbeddingLayer(tf.keras.layers.Layer):
    """Graph embedding layer for summarizing learned information."""
    def __init__(self, filters, name="GEmbed", **kwargs):
        super(GraphEmbeddingLayer, self).__init__(name=name, **kwargs)

        self.num_filters = filters
        self.weight_decay = 0.0005
        self.W = None
        self.b = None

    def build(self, input_shape):
        V_shape = input_shape
        num_features = V_shape[1]
        W_dim = [num_features, self.num_filters]
        b_dim = [self.num_filters]
        W_stddev = 1.0 / math.sqrt(num_features)

        self.W = self.add_weight(
            shape=W_dim,
            dtype=tf.float32,
            initializer=tf.keras.initializers.TruncatedNormal(stddev=W_stddev),
            regularizer=tf.keras.regularizers.l2(self.weight_decay),
            trainable=True,
            name="weight")

        self.b = self.add_weight(
            shape=b_dim,
            dtype=tf.float32,
            initializer=tf.keras.initializers.Constant(0.1),
            trainable=True,
            name="bias")

    def call(self, V, training=None):
        output = tf.matmul(V, self.W) + self.b
        return output


class GraphPoolingLayer(tf.keras.layers.Layer):
    """Graph pooling layer for decreasing number of vertices in graph signal."""
    def __init__(self, num_vertices=1, name="GraphPool", **kwargs):
        super(GraphPoolingLayer, self).__init__(name=name, **kwargs)

        self.num_vertices = num_vertices
        self.weight_decay = 0.0005
        self.W = None
        self.b = None
        self.num_features = None

    def build(self, input_shape):
        V_shape = input_shape
        self.num_features = V_shape[1]
        W_dim = [self.num_features, self.num_vertices]
        b_dim = [self.num_vertices]
        W_stddev = 1.0 / math.sqrt(self.num_features)

        self.W = self.add_weight(
            shape=W_dim,
            dtype=tf.float32,
            initializer=tf.keras.initializers.TruncatedNormal(stddev=W_stddev),
            regularizer=tf.keras.regularizers.l2(self.weight_decay),
            trainable=True,
            name="weight")

        self.b = self.add_weight(
            shape=b_dim,
            dtype=tf.float32,
            initializer=tf.keras.initializers.Constant(0.1),
            trainable=True,
            name="bias")

    def call(self, input, training=None):
        V = input
        factors = tf.matmul(V, self.W) + self.b
        factors = tf.nn.softmax(factors)
        result = tf.matmul(factors, V, transpose_a=True)

        if self.num_vertices == 1:
            return tf.reshape(result, [-1, self.num_features])

        return result


class GlobalPoolingLayer(tf.keras.layers.Layer):
    """Global average pooling layer for graph classification."""
    def __init__(self, name="GlobalPool", **kwargs):
        super(GlobalPoolingLayer, self).__init__(name=name, **kwargs)

    def call(self, input, training=None):
        V, I = input
        output = tf.math.segment_mean(V, I)

        return output


class TransferLayer(tf.keras.layers.Layer):
    """Transfer layer for passing learned information between graph levels."""
    def __init__(self, name="Transfer", **kwargs):
        super(TransferLayer, self).__init__(name=name, **kwargs)

        self.weight_decay = 0.0005

    def build(self, input_shape):
        V_shape, V_aux_shape, _ = input_shape

        W_dim = [V_shape[1], V_aux_shape[1]]
        W_stddev = math.sqrt(1.0 / (V_shape[1] * 2 * GraphCNNGlobal.GRAPHCNN_INIT_FACTOR))

        self.W = self.add_weight(
            shape=W_dim,
            dtype=tf.float32,
            initializer=tf.keras.initializers.TruncatedNormal(stddev=W_stddev),
            regularizer=tf.keras.regularizers.l2(self.weight_decay),
            trainable=True,
            name="weight")

    def call(self, input, training=None):
        V, _, A_linkage = input
        n = tf.matmul(A_linkage, V)
        output = tf.matmul(n, self.W)
        return output

