import h5py
import numpy as np
import tensorflow as tf
import os
import glob
from src.file_ops import *
from src.batch_ops import *


class GraphCNNKeys(object):
    TRAIN_SUMMARIES = "train_summaries"
    TEST_SUMMARIES = "test_summaries"


class GraphCNNGlobal(object):
    BN_DECAY = 0.999
    GRAPHCNN_INIT_FACTOR = 1.
    GRAPHCNN_I_FACTOR = 1.0


def dataloader_adj(file_path):
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


def dataloader_edge(file_path):
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


def dataloader_single_feat(file_path):
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

        """
        A_1_idx = np.array(group.get("A_1_idx"))
        A_1_values = np.array(group.get("A_1_values"))
        A_1_shape = np.array(group.get("A_1_shape"))
        A_1_sparse = tf.SparseTensor(A_1_idx, A_1_values, A_1_shape)
        A_1 = tf.Variable(tf.sparse.to_dense(A_1_sparse, default_value=0.), dtype=tf.dtypes.float32, name="A_1")        
        """

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

        I_1 = tf.Variable(np.array(group.get("V_1_idx")), dtype=tf.dtypes.int32, name="I_1")

        yield [V_1, E_1, E_2, E_3, V_2, A_2, A_3, A_4, I_1], labels

    hf.close()


def load_graphs_from_csv(main_dir):
    sub_dirs = [f.path for f in os.scandir(main_dir) if f.is_dir()]

    for sub_dir in sub_dirs:
        base_name = sub_dir[len(main_dir):]
        base_path = sub_dir + "/" + base_name

        if not os.path.exists(base_path + "_facefeature.csv"):
            continue
        if not os.path.exists(base_path + "_facetfeature.csv"):
            continue
        if not os.path.exists(base_path + "_faceadj.csv"):
            continue
        if not os.path.exists(base_path + "_facetadj.csv"):
            continue
        if not os.path.exists(base_path + "_facefacetlink.csv"):
            continue

        V_1_keys, V_1, labels = load_brep_features(base_path)
        V_2_keys, V_2 = load_facet_features(base_path)

        A_1, E_1, E_2, E_3 = load_brep_adj(base_path, V_1_keys)
        A_2 = load_facet_adj(base_path, V_2_keys)
        A_3, A_4 = load_face_facet_link(base_path, V_1_keys, V_2_keys)

        V_1 = normalize_data(V_1)
        V_2 = normalize_data(V_2)

        V_1 = tf.Variable(V_1, dtype=tf.dtypes.float32, name="V_1")
        V_2 = tf.Variable(V_2, dtype=tf.dtypes.float32, name="V_2")

        E_1 = tf.Variable(E_1, dtype=tf.dtypes.float32, name="E_1")
        E_2 = tf.Variable(E_2, dtype=tf.dtypes.float32, name="E_2")
        E_3 = tf.Variable(E_3, dtype=tf.dtypes.float32, name="E_3")

        A_2 = tf.Variable(A_2, dtype=tf.dtypes.float32, name="A_2")
        A_3 = tf.Variable(A_3, dtype=tf.dtypes.float32, name="A_3")
        A_4 = tf.Variable(A_4, dtype=tf.dtypes.float32, name="A_4")

        labels = np.array(labels, dtype=np.int16)

        yield [V_1, E_1, E_2, E_3, V_2, A_2, A_3, A_4], labels