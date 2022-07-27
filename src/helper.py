"""Module containing useful functions such as dataloader generator functions."""

import tensorflow as tf
import h5py
import numpy as np


def face_upvoting(facet_to_face, face_labels_true, facet_preds):
    """Function for face upvoting when segmenting the facets on the mesh level."""
    face_labels = {}
    face_labels_upvote = []

    m, n = facet_to_face.shape

    for i in range(m):
        face_index = np.nonzero(facet_to_face[i])[0][0]

        if face_index not in face_labels:
            face_labels[face_index] = [facet_preds[i]]
        else:
            face_labels[face_index].append(facet_preds[i])

    for key, value in face_labels.items():
        counts = np.bincount(np.array(value))
        face_labels_upvote.append(np.argmax(counts))

    batch_pred = np.array(face_labels_upvote)
    c_faces = np.sum((batch_pred == face_labels_true))
    t_faces = np.size(face_labels_true)

    return c_faces, t_faces


def dataloader_adj(file_path):
    """Load dataset with only adjacency matrix information."""
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

        yield [V_1, A_1, V_2, A_2, A_3], labels

    hf.close()


def dataloader_edge(file_path):
    """Load dataset with edge convexity information."""
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

        yield [V_1, E_1, E_2, E_3, V_2, A_2, A_3], labels

    hf.close()


def dataloader_single_feat(file_path):
    """Load dataset with single machining feature per CAD model."""
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

        I_1 = tf.Variable(np.array(group.get("I_1")), dtype=tf.dtypes.int32, name="I_1")

        yield [V_1, E_1, E_2, E_3, V_2, A_2, A_3, I_1], labels

    hf.close()
