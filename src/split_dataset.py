from src.file_ops import *
from src.batch_ops import *


def load_graphs(main_dir, split_dirs):
    graphs = []
    names = []
    V_1_idx = -1
    V_2_idx = -1

    for split_dir in split_dirs:
        base_name = split_dir[len(main_dir):]
        names.append(base_name)
        base_path = split_dir + "/" + base_name

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

        CAD_name = np.array([[base_name]], dtype='S')
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

    return graphs


def save_to_h5(main_dir, split_name, split_dirs):
    file_name = split_name + '.h5'
    hf = h5py.File(file_name, 'w')

    V_1_keys_group = hf.create_group('V_1_keys')
    V_1_group = hf.create_group('V_1')
    V_2_keys_group = hf.create_group('V_2_keys')
    V_2_group = hf.create_group('V_2')
    A_1_group = hf.create_group('A_1')
    A_2_group = hf.create_group('A_2')
    A_3_group = hf.create_group('A_3')
    A_4_group = hf.create_group('A_4')
    E_1_group = hf.create_group('E_1')
    E_2_group = hf.create_group('E_2')
    E_3_group = hf.create_group('E_3')
    labels_group = hf.create_group('labels')

    names = []

    for split_dir in split_dirs:
        base_name = split_dir[len(main_dir) + 1:]
        names.append(base_name)
        base_path = split_dir + "/" + base_name

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

        V_1_keys_group.create_dataset(base_name, data=np.array(V_1_keys, dtype=np.int32))
        V_1_group.create_dataset(base_name, data=V_1)
        labels_group.create_dataset(base_name, data=labels)
        V_2_keys_group.create_dataset(base_name, data=np.array(V_2_keys, dtype=np.int32))
        V_2_group.create_dataset(base_name, data=V_2)
        A_1, E_1, E_2, E_3 = load_brep_adj(base_path, V_1_keys)

        A_1_group.create_dataset(base_name, data=A_1)
        E_1_group.create_dataset(base_name, data=E_1)
        E_2_group.create_dataset(base_name, data=E_2)
        E_3_group.create_dataset(base_name, data=E_3)
        A_2_group.create_dataset(base_name, data=load_facet_adj(base_path, V_2_keys))
        A_3, A_4 = load_face_facet_link(base_path, V_1_keys, V_2_keys)
        A_3_group.create_dataset(base_name, data=A_3)
        A_4_group.create_dataset(base_name, data=A_4)

    hf.create_dataset('names', data=np.array(names, dtype='S'))

    hf.close()


if __name__ == '__main__':
    directory = "CSVs_Single_Feature"
    split = {"train": 0.7, "val": 0.15, "test": 0.15}

    train_dirs, val_dirs, test_dirs = get_split(directory, split)

    save_to_h5(directory, "train", train_dirs)
    save_to_h5(directory, "val", val_dirs)
    save_to_h5(directory, "test", test_dirs)