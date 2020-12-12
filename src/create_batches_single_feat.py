import h5py
import numpy as np


def normalize_data(data, hasFaceType=False):
    data_max = np.max(data, axis=0)
    data_min = np.min(data, axis=0)

    if hasFaceType:
        data_max[-1] = 10
        data_min[-1] = 0

    data_norm = (data - data_min) / (data_max - data_min)
    return data_norm


def disjoint_adj(m1, m2):
    shape_m1 = np.shape(m1)
    shape_m2 = np.shape(m2)

    m3 = np.zeros((shape_m1[0] + shape_m2[0], shape_m1[1] + shape_m2[1]))
    m3[:shape_m1[0], :shape_m1[1]] = m1
    m3[shape_m1[0]:, shape_m1[1]:] = m2
    return m3


def load_dataset_from_h5(file_path, file_name):
    graphs = []
    V_1_idx = -1
    V_2_idx = -1

    path = file_path + file_name + ".h5"
    hf = h5py.File(path, 'r')

    V_1_keys_group = hf.get('V_1_keys')
    V_1_group = hf.get('V_1')
    A_1_group = hf.get('A_1')
    V_2_keys_group = hf.get('V_2_keys')
    V_2_group = hf.get('V_2')
    A_2_group = hf.get('A_2')
    A_3_group = hf.get('A_3')
    A_4_group = hf.get('A_4')
    E_1_group = hf.get('E_1')
    E_2_group = hf.get('E_2')
    E_3_group = hf.get('E_3')
    labels_group = hf.get('labels')

    names = np.array(list(V_1_group.items()))[:, 0]

    for name in names:
        V_1_keys = np.array(V_1_keys_group.get(name))
        V_1 = np.array(V_1_group.get(name))
        A_1 = np.array(A_1_group.get(name))
        V_2_keys = np.array(V_2_keys_group.get(name))
        V_2 = np.array(V_2_group.get(name))
        A_2 = np.array(A_2_group.get(name))
        A_3 = np.array(A_3_group.get(name))
        A_4 = np.array(A_4_group.get(name))
        E_1 = np.array(E_1_group.get(name))
        E_2 = np.array(E_2_group.get(name))
        E_3 = np.array(E_3_group.get(name))
        labels = np.array(labels_group.get(name))

        V_1 = normalize_data(V_1, hasFaceType=True)
        V_2 = normalize_data(V_2)

        CAD_name = np.array([[name]], dtype='S')
        V_1_idx += len(V_1_keys)
        V_2_idx += len(V_2_keys)

        # Index shows where the individual graphs are located in minibatch
        idx = np.array([[V_1_idx, V_2_idx]])

        V_1_keys = np.array(V_1_keys, dtype=np.int32)
        V_2_keys = np.array(V_2_keys, dtype=np.int32)

        graph = {"names": CAD_name, "idx": idx, "V_1_keys": V_1_keys, "V_1": V_1, "A_1": A_1,
                 "V_2_keys": V_2_keys, "V_2": V_2, "A_2": A_2, "A_3": A_3, "A_4": A_4,
                 "E_1": E_1, "E_2": E_2, "E_3": E_3, "labels": labels}

        graphs.append(graph)

    hf.close()

    return graphs


def add_graph_to_batch(batch, graph_sample, idx_count):
    if len(batch["V_1"]) == 0:
        for key in batch.keys():
            if key == "labels":
                label = np.array(np.min(graph_sample[key]), ndmin=1)
                batch[key] = label
            elif key == "V_1_idx":
                batch[key] = np.zeros(len(graph_sample["V_1_keys"]))
            elif key == "V_2_idx":
                batch[key] = np.zeros(len(graph_sample["V_2_keys"]))
            else:
                batch[key] = graph_sample[key]

    else:
        batch["names"] = np.append(batch["names"], graph_sample["names"], axis=0)
        batch["V_1_keys"] = np.append(batch["V_1_keys"], graph_sample["V_1_keys"], axis=0)
        batch["V_1"] = np.append(batch["V_1"], graph_sample["V_1"], axis=0)
        batch["V_2_keys"] = np.append(batch["V_2_keys"], graph_sample["V_2_keys"], axis=0)
        batch["V_2"] = np.append(batch["V_2"], graph_sample["V_2"], axis=0)

        batch["A_1"] = disjoint_adj(batch["A_1"], graph_sample["A_1"])
        batch["A_2"] = disjoint_adj(batch["A_2"], graph_sample["A_2"])
        batch["A_3"] = disjoint_adj(batch["A_3"], graph_sample["A_3"])
        batch["A_4"] = disjoint_adj(batch["A_4"], graph_sample["A_4"])

        batch["E_1"] = disjoint_adj(batch["E_1"], graph_sample["E_1"])
        batch["E_2"] = disjoint_adj(batch["E_2"], graph_sample["E_2"])
        batch["E_3"] = disjoint_adj(batch["E_3"], graph_sample["E_3"])

        label = np.array(np.min(graph_sample["labels"]), ndmin=1)
        batch["labels"] = np.append(batch["labels"], [label])

        V_1_idx = np.full(len(graph_sample["V_1_keys"]), idx_count)
        batch["V_1_idx"] = np.append(batch["V_1_idx"], V_1_idx, axis=0)

        V_2_idx = np.full(len(graph_sample["V_2_keys"]), idx_count)
        batch["V_2_idx"] = np.append(batch["V_2_idx"], V_2_idx, axis=0)

    return batch


def graph_batch_from_graph_list(graph_list, file_path, file_name, graphs_per_batch=32):
    graph_counter = 0
    batch_counter = 0

    np.random.shuffle(graph_list)

    for graph in graph_list:
        if graph_counter == 0:
            raw_batch = {"names": [], "V_1_idx": [], "V_1_keys": [], "V_1": [], "A_1": [], "E_1": [], "E_2": [],
                         "E_3": [], "V_2_idx": [], "V_2_keys": [], "V_2": [], "A_2": [], "A_3": [], "A_4": [],
                         "labels": []}

        if graph_counter >= graphs_per_batch:
            write_batches_to_file_sparse(batch_counter, raw_batch, file_path, file_name)
            graph_counter = 0
            batch_counter += 1

        else:
            raw_batch = add_graph_to_batch(raw_batch, graph, graph_counter)
            graph_counter += 1

    #write_batches_to_file_sparse(batch_counter, raw_batch, file_path, file_name)


def get_sparse_tensor_info(matrix, default_val):
    idx = np.where(np.not_equal(matrix, default_val))
    values = matrix[idx]
    shape = np.shape(matrix)

    idx = np.transpose(idx).astype(np.int32)
    values = values.astype(np.float32)
    shape = np.array(shape).astype(np.int32)

    return [idx, values, shape]


def write_batches_to_file_sparse(batch_num, batch, file_path, file_name):
    path = file_path + file_name + "_sparse.h5"
    hf = h5py.File(path, 'a')

    default_value = 0.
    A_1_data = get_sparse_tensor_info(batch["A_1"], default_value)
    E_1_data = get_sparse_tensor_info(batch["E_1"], default_value)
    E_2_data = get_sparse_tensor_info(batch["E_2"], default_value)
    E_3_data = get_sparse_tensor_info(batch["E_3"], default_value)
    A_2_data = get_sparse_tensor_info(batch["A_2"], default_value)
    A_3_data = get_sparse_tensor_info(batch["A_3"], default_value)
    A_4_data = get_sparse_tensor_info(batch["A_4"], default_value)

    batch_group = hf.create_group(str(batch_num))

    batch_group.create_dataset("CAD_model", data=batch["names"], compression="gzip", compression_opts=9)
    batch_group.create_dataset("V_1_keys", data=batch["V_1_keys"], compression="gzip", compression_opts=9)
    batch_group.create_dataset("V_1_idx", data=batch["V_1_idx"])
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

    batch_group.create_dataset("V_2_keys", data=batch["V_2_keys"], compression="gzip", compression_opts=9)
    batch_group.create_dataset("V_2_idx", data=batch["V_2_idx"])
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


if __name__ == '__main__':
    f_path = "/home/mlg/Documents/Andrew/hierarchical-cadnet/data/Single_Feature_70_15_15/"

    print("Processing Training Set")
    f_name = "train"
    graph_list = load_dataset_from_h5(f_path, f_name)
    graph_batch_from_graph_list(graph_list, f_path, f_name, graphs_per_batch=64)

    print("Processing Validation Set")
    f_name = "val"
    graph_list = load_dataset_from_h5(f_path, f_name)
    graph_batch_from_graph_list(graph_list, f_path, f_name)    

    print("Processing Test Set")
    f_name = "test"
    graph_list = load_dataset_from_h5(f_path, f_name)
    graph_batch_from_graph_list(graph_list, f_path, f_name, graphs_per_batch=64)
