import h5py
import numpy as np
import tensorflow as tf


class GraphCNNKeys(object):
    TRAIN_SUMMARIES = "train_summaries"
    TEST_SUMMARIES = "test_summaries"


class GraphCNNGlobal(object):
    BN_DECAY = 0.999
    GRAPHCNN_INIT_FACTOR = 1.
    GRAPHCNN_I_FACTOR = 1.0


def h5_dataloader(file_path):
    hf = h5py.File(file_path, 'r')

    for key in list(hf.keys()):
        group = hf.get(key)

        V_1 = tf.Variable(np.array(group.get("V_1"), dtype=np.float32), dtype=tf.float32, name="V_1")
        A_1 = tf.Variable(np.array(group.get("A_1"), dtype=np.float32), dtype=tf.float32, name="A_1")
        V_2 = tf.Variable(np.array(group.get("V_2"), dtype=np.float32), dtype=tf.float32, name="V_2")
        A_2 = tf.Variable(np.array(group.get("A_2"), dtype=np.float32), dtype=tf.float32, name="A_2")
        A_3 = tf.Variable(np.array(group.get("A_3"), dtype=np.float32), dtype=tf.float32, name="A_3")
        A_4 = tf.Variable(np.array(group.get("A_4"), dtype=np.float32), dtype=tf.float32, name="A_4")
        labels = np.array(group.get("labels"), dtype=np.int8)

        yield [V_1, A_1, V_2, A_2, A_3, A_4], labels

    hf.close()


def dataloader_sparse(file_path):
    hf = h5py.File(file_path, 'r')

    for key in list(hf.keys()):
        group = hf.get(key)

        V_1 = tf.Variable(np.array(group.get("V_1")), dtype=tf.dtypes.float32, name="V_1")
        V_2 = tf.Variable(np.array(group.get("V_2")), dtype=tf.dtypes.float32, name="V_2")
        labels = np.array(group.get("labels"), dtype=np.int16)

        A_1_idx = np.array(group.get("A_1_idx"))
        A_1_values = np.array(group.get("A_1_values"))
        A_1_shape = np.array(group.get("A_1_shape"))
        A_1_sparse = tf.SparseTensor(A_1_idx, A_1_values, A_1_shape)
        A_1 = tf.Variable(tf.sparse.to_dense(A_1_sparse, default_value=0.), dtype=tf.dtypes.float32, name="A_1")

        A_2_idx = np.array(group.get("A_2_idx"))
        A_2_values = np.array(group.get("A_2_values"))
        A_2_shape = np.array(group.get("A_2_shape"))
        A_2_sparse = tf.SparseTensor(A_2_idx, A_2_values, A_2_shape)
        A_2 = tf.Variable(tf.sparse.to_dense(A_2_sparse, default_value=0.), dtype=tf.dtypes.float32, name="A_2")

        A_3_idx = np.array(group.get("A_3_idx"))
        A_3_values = np.array(group.get("A_3_values"))
        A_3_shape = np.array(group.get("A_3_shape"))
        A_3_sparse = tf.SparseTensor(A_3_idx, A_3_values, A_3_shape)
        A_3 = tf.Variable(tf.sparse.to_dense(A_3_sparse, default_value=0.), dtype=tf.dtypes.float32, name="A_3")

        A_4_idx = np.array(group.get("A_4_idx"))
        A_4_values = np.array(group.get("A_4_values"))
        A_4_shape = np.array(group.get("A_4_shape"))
        A_4_sparse = tf.SparseTensor(A_4_idx, A_4_values, A_4_shape)
        A_4 = tf.Variable(tf.sparse.to_dense(A_4_sparse, default_value=0.), dtype=tf.dtypes.float32, name="A_4")

        yield [V_1, A_1, V_2, A_2, A_3, A_4], labels

    hf.close()


def dataloader_edge_sparse(file_path):
    hf = h5py.File(file_path, 'r')

    for key in list(hf.keys()):
        group = hf.get(key)

        V_1 = tf.Variable(np.array(group.get("V_1")), dtype=tf.dtypes.float32, name="V_1")
        V_2 = tf.Variable(np.array(group.get("V_2")), dtype=tf.dtypes.float32, name="V_2")
        labels = np.array(group.get("labels"), dtype=np.int16)

        E_1_idx = np.array(group.get("E_1_idx"))
        E_1_values = np.array(group.get("E_1_values"))
        E_1_shape = np.array(group.get("E_1_shape"))
        E_1_sparse = tf.SparseTensor(E_1_idx, E_1_values, E_1_shape)
        E_1 = tf.Variable(tf.sparse.to_dense(E_1_sparse, default_value=0.), dtype=tf.dtypes.float32, name="E_1")

        E_2_idx = np.array(group.get("E_2_idx"))
        E_2_values = np.array(group.get("E_2_values"))
        E_2_shape = np.array(group.get("E_2_shape"))
        E_2_sparse = tf.SparseTensor(E_2_idx, E_2_values, E_2_shape)
        E_2 = tf.Variable(tf.sparse.to_dense(E_2_sparse, default_value=0.), dtype=tf.dtypes.float32, name="E_2")

        E_3_idx = np.array(group.get("E_3_idx"))
        E_3_values = np.array(group.get("E_3_values"))
        E_3_shape = np.array(group.get("E_3_shape"))
        E_3_sparse = tf.SparseTensor(E_3_idx, E_3_values, E_3_shape)
        E_3 = tf.Variable(tf.sparse.to_dense(E_3_sparse, default_value=0.), dtype=tf.dtypes.float32, name="E_3")

        A_2_idx = np.array(group.get("A_2_idx"))
        A_2_values = np.array(group.get("A_2_values"))
        A_2_shape = np.array(group.get("A_2_shape"))
        A_2_sparse = tf.SparseTensor(A_2_idx, A_2_values, A_2_shape)
        A_2 = tf.Variable(tf.sparse.to_dense(A_2_sparse, default_value=0.), dtype=tf.dtypes.float32, name="A_2")

        A_3_idx = np.array(group.get("A_3_idx"))
        A_3_values = np.array(group.get("A_3_values"))
        A_3_shape = np.array(group.get("A_3_shape"))
        A_3_sparse = tf.SparseTensor(A_3_idx, A_3_values, A_3_shape)
        A_3 = tf.Variable(tf.sparse.to_dense(A_3_sparse, default_value=0.), dtype=tf.dtypes.float32, name="A_3")

        A_4_idx = np.array(group.get("A_4_idx"))
        A_4_values = np.array(group.get("A_4_values"))
        A_4_shape = np.array(group.get("A_4_shape"))
        A_4_sparse = tf.SparseTensor(A_4_idx, A_4_values, A_4_shape)
        A_4 = tf.Variable(tf.sparse.to_dense(A_4_sparse, default_value=0.), dtype=tf.dtypes.float32, name="A_4")

        yield [V_1, E_1, E_2, E_3, V_2, A_2, A_3, A_4], labels

    hf.close()


def dataloader_sparse_level_3(file_path):
    hf = h5py.File(file_path, 'r')

    for key in list(hf.keys()):
        group = hf.get(key)

        V_1 = tf.Variable(np.array(group.get("V_1")), dtype=tf.dtypes.float32, name="V_1")
        V_2 = tf.Variable(np.array(group.get("V_2")), dtype=tf.dtypes.float32, name="V_2")
        V_3 = tf.Variable(np.array(group.get("V_3")), dtype=tf.dtypes.float32, name="V_3")
        labels = np.array(group.get("labels"))

        A_1_idx = np.array(group.get("A_1_idx"))
        A_1_values = np.array(group.get("A_1_values"))
        A_1_shape = np.array(group.get("A_1_shape"))
        A_1_sparse = tf.SparseTensor(A_1_idx, A_1_values, A_1_shape)
        A_1 = tf.Variable(tf.sparse.to_dense(A_1_sparse, default_value=0.), dtype=tf.dtypes.float32, name="A_1")

        A_2_idx = np.array(group.get("A_2_idx"))
        A_2_values = np.array(group.get("A_2_values"))
        A_2_shape = np.array(group.get("A_2_shape"))
        A_2_sparse = tf.SparseTensor(A_2_idx, A_2_values, A_2_shape)
        A_2 = tf.Variable(tf.sparse.to_dense(A_2_sparse, default_value=0.), dtype=tf.dtypes.float32, name="A_2")

        A_3_idx = np.array(group.get("A_3_idx"))
        A_3_values = np.array(group.get("A_3_values"))
        A_3_shape = np.array(group.get("A_3_shape"))
        A_3_sparse = tf.SparseTensor(A_3_idx, A_3_values, A_3_shape)
        A_3 = tf.Variable(tf.sparse.to_dense(A_3_sparse, default_value=0.), dtype=tf.dtypes.float32, name="A_3")

        A_4_idx = np.array(group.get("A_4_idx"))
        A_4_values = np.array(group.get("A_4_values"))
        A_4_shape = np.array(group.get("A_4_shape"))
        A_4_sparse = tf.SparseTensor(A_4_idx, A_4_values, A_4_shape)
        A_4 = tf.Variable(tf.sparse.to_dense(A_4_sparse, default_value=0.), dtype=tf.dtypes.float32, name="A_4")

        A_5_idx = np.array(group.get("A_5_idx"))
        A_5_values = np.array(group.get("A_5_values"))
        A_5_shape = np.array(group.get("A_5_shape"))
        A_5_sparse = tf.SparseTensor(A_5_idx, A_5_values, A_5_shape)
        A_5 = tf.Variable(tf.sparse.to_dense(A_5_sparse, default_value=0.), dtype=tf.dtypes.float32, name="A_5")

        A_6_idx = np.array(group.get("A_6_idx"))
        A_6_values = np.array(group.get("A_6_values"))
        A_6_shape = np.array(group.get("A_6_shape"))
        A_6_sparse = tf.SparseTensor(A_6_idx, A_6_values, A_6_shape)
        A_6 = tf.Variable(tf.sparse.to_dense(A_6_sparse, default_value=0.), dtype=tf.dtypes.float32, name="A_6")

        A_7_idx = np.array(group.get("A_7_idx"))
        A_7_values = np.array(group.get("A_7_values"))
        A_7_shape = np.array(group.get("A_7_shape"))
        A_7_sparse = tf.SparseTensor(A_7_idx, A_7_values, A_7_shape)
        A_7 = tf.Variable(tf.sparse.to_dense(A_7_sparse, default_value=0.), dtype=tf.dtypes.float32, name="A_7")

        yield [V_1, A_1, V_2, A_2, A_3, A_4, V_3, A_5, A_6, A_7], labels

    hf.close()