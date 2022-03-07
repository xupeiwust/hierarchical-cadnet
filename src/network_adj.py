"""Module for defining Hierarchical CADNet network architecture using only adjacency information."""

from src.layers import *


class HierarchicalGCNN(tf.keras.Model):

    def __init__(self, units, rate, num_classes, num_layers=7):
        super(HierarchicalGCNN, self).__init__()
        self.num_layers = num_layers
        # Labelling Code: nnlayer_level_block
        self.ge_start = GraphEmbeddingLayer(filters=units, name="GE_start")
        self.bn_start = tf.keras.layers.BatchNormalization(name="BN_start")
        self.dp_start = tf.keras.layers.Dropout(rate=rate, name="DP_start")

        for i in range(1, self.num_layers + 1):
            setattr(self, f"gcnn_1_{i}", GraphCNNLayer(filters=units, name=f"GCNN_1_{i}"))
            setattr(self, f"bn_1_{i}", tf.keras.layers.BatchNormalization(name=f"BN_1_{i}"))
            setattr(self, f"dp_1_{i}", tf.keras.layers.Dropout(rate=rate, name=f"DP_1_{i}"))

        for i in range(1, self.num_layers + 1):
            setattr(self, f"gcnn_2_{i}", GraphCNNLayer(filters=units, name=f"GCNN_2_{i}"))
            setattr(self, f"bn_2_{i}", tf.keras.layers.BatchNormalization(name=f"BN_2_{i}"))
            setattr(self, f"dp_2_{i}", tf.keras.layers.Dropout(rate=rate, name=f"DP_2_{i}"))

        self.ge_1 = GraphEmbeddingLayer(filters=units, name="GE_1")
        self.bn_1 = tf.keras.layers.BatchNormalization(name="BN_1")
        self.dp_1 = tf.keras.layers.Dropout(rate=rate, name="DP_1")

        self.ge_2 = GraphEmbeddingLayer(filters=units, name="GE_2")
        self.bn_2 = tf.keras.layers.BatchNormalization(name="BN_2")
        self.dp_2 = tf.keras.layers.Dropout(rate=rate, name="DP_2")

        # Transfer Layers
        self.a3 = TransferLayer(name="A3")
        self.bn_a3 = tf.keras.layers.BatchNormalization(name="BN_A3")
        self.a4 = TransferLayer(name="A4")
        self.bn_a4 = tf.keras.layers.BatchNormalization(name="BN_A4")

        # Level 1 - Final
        self.ge_final = GraphEmbeddingLayer(filters=num_classes, name="GE_final")
        self.softmax = tf.keras.layers.Softmax()

    def call(self, inputs, training=False):
        V_1, A_1, V_2, A_2, A_3 = inputs

        x_1 = self.ge_start(V_1)
        x_1 = self.bn_start(x_1, training=training)
        x_1 = tf.nn.relu(x_1)
        x_1 = self.dp_start(x_1, training=training)

        # A4 => Projection from Level 1 to Level 2
        a_4 = self.a4([x_1, V_2, A_3])
        a_4 = self.bn_a4(a_4, training=training)
        a_4 = tf.nn.relu(a_4)
        x_2 = V_2 + a_4

        for i in range(1, self.num_layers + 1):
            r_2 = getattr(self, f"gcnn_2_{i}")([x_2, A_2])
            r_2 = getattr(self, f"bn_2_{i}")(r_2, training=training)
            r_2 = tf.nn.relu(r_2)
            r_2 = getattr(self, f"dp_2_{i}")(r_2, training=training)

            if i == 1:
                x_2 = r_2
            else:
                x_2 += r_2

        x_2 = self.ge_2(x_2)
        x_2 = self.bn_2(x_2, training=training)
        x_2 = tf.nn.relu(x_2)
        x_2 = self.dp_2(x_2, training=training)

        # A3 => Embedding from Level 2 to Level 1
        a_3 = self.a3([x_2, x_1, tf.transpose(A_3)])
        a_3 = self.bn_a3(a_3, training=training)
        a_3 = tf.nn.relu(a_3)
        x_1 += a_3

        for i in range(1, self.num_layers + 1):
            r_1 = getattr(self, f"gcnn_1_{i}")([x_1, A_1])
            r_1 = getattr(self, f"bn_1_{i}")(r_1, training=training)
            r_1 = tf.nn.relu(r_1)
            r_1 = getattr(self, f"dp_1_{i}")(r_1, training=training)
            x_1 += r_1

        x_1 = self.ge_1(x_1)
        x_1 = self.bn_1(x_1, training=training)
        x_1 = tf.nn.relu(x_1)
        x_1 = self.dp_1(x_1, training=training)

        # Final 0-hop layer
        x = self.ge_final(x_1)
        x = self.softmax(x)

        return x