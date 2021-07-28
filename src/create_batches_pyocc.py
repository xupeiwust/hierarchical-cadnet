import h5py
import random
import glob
import ntpath
import math
import numpy as np
from scipy.sparse import csr_matrix


def normalize_data(data, has_face_type=True):
    epilson = 1e-5
    data_max = np.max(data, axis=0)
    data_min = np.min(data, axis=0)

    if has_face_type:
        data_max[-1] = 11
        data_min[-1] = 0

    data_norm = (data - data_min) / (data_max - data_min + epilson)
    return data_norm


def disjoint_adj(m1, m2):
    shape_m1 = np.shape(m1)
    shape_m2 = np.shape(m2)

    m3 = np.zeros((shape_m1[0] + shape_m2[0], shape_m1[1] + shape_m2[1]))
    m3[:shape_m1[0], :shape_m1[1]] = m1
    m3[shape_m1[0]:, shape_m1[1]:] = m2
    return m3


def get_sparse_tensor_info(matrix, default_val):
    idx = np.where(np.not_equal(matrix, default_val))
    values = matrix[idx]
    shape = np.shape(matrix)

    idx = np.transpose(idx).astype(np.int32)
    values = values.astype(np.float32)
    shape = np.array(shape).astype(np.int32)

    return [idx, values, shape]


def add_graph_to_batch(batch, graph_sample):
    if len(batch["V_1"]) == 0:
        for key in batch.keys():
            batch[key] = graph_sample[key]

    else:
        batch["names"] = np.append(batch["names"], graph_sample["names"], axis=0)
        batch["idx"] = np.append(batch["idx"], graph_sample["idx"], axis=0)
        batch["V_1"] = np.append(batch["V_1"], graph_sample["V_1"], axis=0)
        batch["V_2"] = np.append(batch["V_2"], graph_sample["V_2"], axis=0)
        batch["labels"] = np.append(batch["labels"], graph_sample["labels"], axis=0)

        batch["A_1"] = disjoint_adj(batch["A_1"], graph_sample["A_1"])
        batch["A_2"] = disjoint_adj(batch["A_2"], graph_sample["A_2"])
        batch["A_3"] = disjoint_adj(batch["A_3"], graph_sample["A_3"])
        batch["A_4"] = disjoint_adj(batch["A_4"], graph_sample["A_4"])

        batch["E_1"] = disjoint_adj(batch["E_1"], graph_sample["E_1"])
        batch["E_2"] = disjoint_adj(batch["E_2"], graph_sample["E_2"])
        batch["E_3"] = disjoint_adj(batch["E_3"], graph_sample["E_3"])

    return batch


def graph_batch_from_graph_list(graph_list, file_path, nodes_per_batch=10000):
    node_counter = 0
    batch_counter = 0

    for graph in graph_list:
        if node_counter == 0:
            raw_batch = {"names": [], "idx": [], "V_1": [], "A_1": [], "E_1": [], "E_2": [], "E_3": [], "V_2": [],
                         "A_2": [], "A_3": [], "A_4": [], "labels": []}

        node_counter += len(graph["V_1"]) + len(graph["V_2"])

        if node_counter >= nodes_per_batch:
            node_counter = 0

            try:
                write_batches_to_file(batch_counter, raw_batch, file_path)
                batch_counter += 1
            except Exception as e:
                print(e)
                continue
        else:
            raw_batch = add_graph_to_batch(raw_batch, graph)

    try:
        write_batches_to_file(batch_counter, raw_batch, file_path)
    except Exception as e:
        print(e)


def write_batches_to_file(batch_num, batch, file_path):
    hf = h5py.File(file_path, 'a')

    default_value = 0.
    A_1_data = get_sparse_tensor_info(batch["A_1"], default_value)
    E_1_data = get_sparse_tensor_info(batch["E_1"], default_value)
    E_2_data = get_sparse_tensor_info(batch["E_2"], default_value)
    E_3_data = get_sparse_tensor_info(batch["E_3"], default_value)
    A_2_data = get_sparse_tensor_info(batch["A_2"], default_value)
    A_3_data = get_sparse_tensor_info(batch["A_3"], default_value)
    A_4_data = get_sparse_tensor_info(batch["A_4"], default_value)

    batch_group = hf.create_group(str(batch_num))

    batch_group.create_dataset("CAD_model", data=np.array(batch["names"], dtype="S"), compression="gzip", compression_opts=9)
    batch_group.create_dataset("idx", data=batch["idx"], compression="gzip", compression_opts=9)
    batch_group.create_dataset("V_1", data=batch["V_1"])

    batch_group.create_dataset("A_1_idx", data=A_1_data[0])
    batch_group.create_dataset("A_1_values", data=A_1_data[1])
    batch_group.create_dataset("A_1_shape", data=A_1_data[2])

    batch_group.create_dataset("E_1_idx", data=E_1_data[0])
    batch_group.create_dataset("E_1_values", data=E_1_data[1])
    batch_group.create_dataset("E_1_shape", data=E_1_data[2])
    batch_group.create_dataset("E_2_idx", data=E_2_data[0])
    batch_group.create_dataset("E_2_values", data=E_2_data[1])
    batch_group.create_dataset("E_2_shape", data=E_2_data[2])
    batch_group.create_dataset("E_3_idx", data=E_3_data[0])
    batch_group.create_dataset("E_3_values", data=E_3_data[1])
    batch_group.create_dataset("E_3_shape", data=E_3_data[2])

    batch_group.create_dataset("V_2", data=batch["V_2"])
    batch_group.create_dataset("A_2_idx", data=A_2_data[0])
    batch_group.create_dataset("A_2_values", data=A_2_data[1])
    batch_group.create_dataset("A_2_shape", data=A_2_data[2])
    batch_group.create_dataset("A_3_idx", data=A_3_data[0])
    batch_group.create_dataset("A_3_values", data=A_3_data[1])
    batch_group.create_dataset("A_3_shape", data=A_3_data[2])
    batch_group.create_dataset("A_4_idx", data=A_4_data[0])
    batch_group.create_dataset("A_4_values", data=A_4_data[1])
    batch_group.create_dataset("A_4_shape", data=A_4_data[2])
    batch_group.create_dataset("labels", data=batch["labels"])

    hf.close()


def get_dense_tensor(idxs, values, shape):
    dense_tensor = csr_matrix((values, idxs.T), shape=shape).toarray().astype(np.int32)
    return dense_tensor


def graph_batch_from_graph_generator(graph_generator, file_path, nodes_per_batch=10000):
    node_counter = 0
    batch_counter = 0

    for graph in graph_generator:
        if node_counter == 0:
            raw_batch = {"names": [], "idx": [], "V_1": [], "A_1": [], "E_1": [], "E_2": [], "E_3": [], "V_2": [],
                         "A_2": [], "A_3": [], "A_4": [], "labels": []}

        node_counter += len(graph["V_1"]) + len(graph["V_2"])

        if node_counter >= nodes_per_batch:
            node_counter = 0
            try:
                write_batches_to_file(batch_counter, raw_batch, file_path)
                batch_counter += 1
            except Exception as e:
                print(e)
                continue

        else:
            raw_batch = add_graph_to_batch(raw_batch, graph)

    try:
        write_batches_to_file(batch_counter, raw_batch, file_path)
    except Exception as e:
        print(e)


def dataset_generator(file_path, ids):
    V_1_idx = -1
    V_2_idx = -1

    hf = h5py.File(file_path, 'r')

    for id in ids:
        group = hf.get(id)
        V_1 = np.array(group.get('V_1'))
        V_2 = np.array(group.get('V_2'))
        labels = np.array(group.get('labels'))

        V_1 = normalize_data(V_1, has_face_type=True)
        V_2 = normalize_data(V_2)

        A_1_idx = np.array(group.get('A_1_idx'))
        A_1_values = np.array(group.get('A_1_values'))
        A_1_shape = np.array(group.get('A_1_shape'))

        E_1_idx = np.array(group.get('E_1_idx'))
        E_1_values = np.array(group.get('E_1_values'))
        E_1_shape = np.array(group.get('E_1_shape'))

        E_2_idx = np.array(group.get('E_2_idx'))
        E_2_values = np.array(group.get('E_2_values'))
        E_2_shape = np.array(group.get('E_2_shape'))

        E_3_idx = np.array(group.get('E_3_idx'))
        E_3_values = np.array(group.get('E_3_values'))
        E_3_shape = np.array(group.get('E_3_shape'))

        A_2_idx = np.array(group.get('A_2_idx'))
        A_2_values = np.array(group.get('A_2_values'))
        A_2_shape = np.array(group.get('A_2_shape'))

        A_3_idx = np.array(group.get('A_3_idx'))
        A_3_values = np.array(group.get('A_3_values'))
        A_3_shape = np.array(group.get('A_3_shape'))

        A_4_idx = np.array(group.get('A_4_idx'))
        A_4_values = np.array(group.get('A_4_values'))
        A_4_shape = np.array(group.get('A_4_shape'))

        A_1 = get_dense_tensor(A_1_idx, A_1_values, A_1_shape)
        A_2 = get_dense_tensor(A_2_idx, A_2_values, A_2_shape)
        A_3 = get_dense_tensor(A_3_idx, A_3_values, A_3_shape)
        A_4 = get_dense_tensor(A_4_idx, A_4_values, A_4_shape)

        E_1 = get_dense_tensor(E_1_idx, E_1_values, E_1_shape)
        E_2 = get_dense_tensor(E_2_idx, E_2_values, E_2_shape)
        E_3 = get_dense_tensor(E_3_idx, E_3_values, E_3_shape)

        V_1_idx += len(V_1)
        V_2_idx += len(V_2)

        # Index shows where the individual graphs are located in minibatch
        idx = np.array([[V_1_idx, V_2_idx]])

        graph = {"names": [id], "idx": idx, "V_1": V_1, "A_1": A_1, "V_2": V_2, "A_2": A_2, "A_3": A_3, "A_4": A_4,
                 "E_1": E_1, "E_2": E_2, "E_3": E_3, "labels": labels}

        yield graph

    hf.close()


def split_dataset(file_path, dataset_split):
    hf = h5py.File(file_path, 'r')
    keys = hf.keys()

    file_ids = list(keys)
    random.shuffle(file_ids)
    random.shuffle(file_ids)
    num_dirs = len(file_ids)

    train_split = int(math.ceil(num_dirs * dataset_split["train"]))
    val_split = int(math.ceil(num_dirs * dataset_split["val"]))

    train_ids = file_ids[:train_split]
    val_ids = file_ids[train_split:train_split + val_split]
    test_ids = file_ids[train_split + val_split:]

    hf.close()

    return train_ids, val_ids, test_ids


def get_split_ids_from_txt_file(train_txt, val_txt, test_txt):
    train_ids = []
    val_ids = []
    test_ids = []

    f = open(train_txt, "r")
    for line in f:
        id = line[:-1]
        train_ids.append(id)
    f.close()

    f = open(val_txt, "r")
    for line in f:
        id = line[:-1]
        val_ids.append(id)
    f.close()

    f = open(test_txt, "r")
    for line in f:
        id = line[:-1] # remove \n
        test_ids.append(id)
    f.close()

    return train_ids, val_ids, test_ids


def get_split_ids_from_dirs(main_dir):
    train_dir = main_dir + "train/"
    val_dir = main_dir + "val/"
    test_dir = main_dir + "test/"

    train_files = glob.glob(train_dir + "*.step")
    val_files = glob.glob(val_dir + "*.step")
    test_files = glob.glob(test_dir + "*.step")

    train_ids = []
    val_ids = []
    test_ids = []

    for file in train_files:
        base_name = ntpath.basename(file)[:-len(".step")]
        train_ids.append(base_name)

    for file in val_files:
        base_name = ntpath.basename(file)[:-len(".step")]
        val_ids.append(base_name)

    for file in test_files:
        base_name = ntpath.basename(file)[:-len(".step")]
        test_ids.append(base_name)

    return train_ids, val_ids, test_ids


if __name__ == '__main__':
    import os
    current_dir_path = os.path.dirname(os.path.realpath(__file__))
    base_dir_path = os.path.dirname(current_dir_path)
    #data_dir_path = os.path.join(base_dir_path, "data") + "/"
    data_dir_path = "/home/mlg/Documents/Andrew/Datasets/MFCAD++/hierarchical_graphs/"

    # Parameters
    # h5_path = data_dir_path + "MFCAD++_dataset.h5"
    h5_path = "/home/mlg/Documents/Andrew/Datasets/MFCAD++/hierarchical_graphs/MFCAD++_dataset.h5"
    max_num_nodes_per_batch = 10000
    split = {"train": 0.7, "val": 0.15, "test": 0.15}

    train_txt_path = "/home/mlg/Documents/Andrew/Datasets/MFCAD++/train.txt"
    val_txt_path = "/home/mlg/Documents/Andrew/Datasets/MFCAD++/val.txt"
    test_txt_path = "/home/mlg/Documents/Andrew/Datasets/MFCAD++/test.txt"

    train_ids, val_ids, test_ids = get_split_ids_from_txt_file(train_txt_path, val_txt_path, test_txt_path)
    # train_ids, val_ids, test_ids = get_split_ids_from_dirs("/home/mlg/Documents/Andrew/UV-Net/data/MFCAD/step/")
    # train_ids, val_ids, test_ids = split_dataset(h5_path, split)

    #write_splits_to_txt_file(data_dir_path, train_ids, val_ids, test_ids)

    print("Processing Training Set")
    train_h5_path = data_dir_path + "training_MFCAD++.h5"
    print("Loading graphs")
    graph_generator = dataset_generator(h5_path, train_ids)
    print("Creating batches")
    graph_batch_from_graph_generator(graph_generator, train_h5_path, max_num_nodes_per_batch)

    print("Processing Validation Set")
    val_h5_path = data_dir_path + "val_MFCAD++.h5"
    print("Loading graphs")
    graph_generator = dataset_generator(h5_path, val_ids)
    print("Creating batches")
    graph_batch_from_graph_generator(graph_generator, val_h5_path, max_num_nodes_per_batch)    

    print("Processing Test Set")
    test_h5_path = data_dir_path + "test_MFCAD++.h5"
    print("Loading graphs")
    graph_generator = dataset_generator(h5_path, test_ids)
    print("Creating batches")
    graph_batch_from_graph_generator(graph_generator, test_h5_path, max_num_nodes_per_batch)    
