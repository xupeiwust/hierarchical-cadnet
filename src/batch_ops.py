import os
import h5py
import random
import numpy as np


def normalize_data(data):
    data_max = np.max(data, axis=0)
    data_min = np.min(data, axis=0)
    data_norm = (data - data_min) / (data_max - data_min)
    return data_norm


def disjoint_adj(m1, m2):
    shape_m1 = np.shape(m1)
    shape_m2 = np.shape(m2)

    m3 = np.zeros((shape_m1[0] + shape_m2[0], shape_m1[1] + shape_m2[1]))
    m3[:shape_m1[0], :shape_m1[1]] = m1
    m3[shape_m1[0]:, shape_m1[1]:] = m2
    return m3


def graph_batch_from_graph_list(graph_list, file_path, file_name, max_nodes_per_batch=15000, num_levels=2, is_sparse=True, edges=True):   
    node_counter = 0
    batch_counter = 0

    for graph in graph_list:
        if node_counter == 0:
            raw_batch = {"names": [], "idx": [], "labels": []}
            adj_count = 2
            
            for level in range(1, num_levels+1):
                node_counter += len(graph[f"V_{level}"])
                raw_batch[f"V_{level}_keys"] = []
                raw_batch[f"V_{level}"] = []
                
                if level != 1:
                    raw_batch[f"A_{adj_count}"] = []
                    adj_count+=1
                    raw_batch[f"A_{adj_count}"] = []
                    adj_count+=1 
                    raw_batch[f"A_{adj_count}"] = []
                    adj_count+=1 
                    
            if edges:
                raw_batch["E_1"] = []
                raw_batch["E_2"] = []
        
        if node_counter > max_nodes_per_batch:
            node_counter = 0
            if is_sparse:
                write_batches_to_file_sparse(batch_counter, raw_batch, file_path, file_name, num_levels)
            else:
                write_batches_to_file(batch_counter, raw_batch, file_path, file_name, num_levels)
            batch_counter += 1
                  
        else:
            raw_batch = add_graph_to_batch(raw_batch, graph, num_levels)
            
    if is_sparse:
        write_batches_to_file_sparse(batch_counter, raw_batch, file_path, file_name, num_levels)
    else:
        write_batches_to_file(batch_counter, raw_batch, file_path, file_name, num_levels)


def get_sparse_tensor_info(matrix, default_val):
    idx = np.where(np.not_equal(matrix, default_val))
    values = matrix[idx]
    shape = np.shape(matrix)
    
    idx = np.transpose(idx).astype(np.int32)
    values = values.astype(np.float32)
    shape = np.array(shape).astype(np.int32)
    
    return [idx, values, shape]


def write_batches_to_file(batch_num, batch, file_path, file_name, num_levels=2):
    path = file_path + file_name + "_dense.h5"
    hf = h5py.File(path, 'a')

    batch_group = hf.create_group(str(batch_num))
    batch_group.create_dataset("CAD_model", data=batch["names"], compression="gzip", compression_opts=9)
    batch_group.create_dataset("idx", data=batch["idx"], compression="gzip", compression_opts=9)
    
    adj_count = 2
    
    for level in range(1, num_levels+1):
        batch_group.create_dataset(f"V_{level}_keys", data=batch[f"V_{level}_keys"], compression="gzip", compression_opts=9)
        batch_group.create_dataset(f"V_{level}", data=batch[f"V_{level}"])
        
        if level == 1:
            batch_group.create_dataset("labels", data=batch["labels"])    
        else:
            batch_group.create_dataset(f"A_{adj_count}", data=batch[f"A_{adj_count}"], compression="lzf")
            adj_count += 1
            batch_group.create_dataset(f"A_{adj_count}", data=batch[f"A_{adj_count}"], compression="lzf")
            adj_count += 1
            batch_group.create_dataset(f"A_{adj_count}", data=batch[f"A_{adj_count}"], compression="lzf")  
            adj_count += 1

    hf.close()


def write_batches_to_file_sparse(batch_num, batch, file_path, file_name, num_levels=2, edges=True):
    path = file_path + file_name + "_sparse.h5"
    hf = h5py.File(path, 'a')
    
    batch_group = hf.create_group(str(batch_num))
    batch_group.create_dataset("CAD_model", data=batch["names"], compression="gzip", compression_opts=9)
    batch_group.create_dataset("idx", data=batch["idx"], compression="gzip", compression_opts=9)
    
    default_value = 0.
    adj_count = 2
    
    for level in range(1, num_levels+1):
        batch_group.create_dataset(f"V_{level}_keys", data=batch[f"V_{level}_keys"], compression="gzip", compression_opts=9)
        batch_group.create_dataset(f"V_{level}", data=batch[f"V_{level}"])
        
        if level == 1:
            batch_group.create_dataset("labels", data=batch["labels"])    
        else:
            A_data = get_sparse_tensor_info(batch[f"A_{adj_count}"], default_value)
            batch_group.create_dataset(f"A_{adj_count}_idx", data=A_data[0])
            batch_group.create_dataset(f"A_{adj_count}_values", data=A_data[1])
            batch_group.create_dataset(f"A_{adj_count}_shape", data=A_data[2])
            adj_count += 1
        
            A_data = get_sparse_tensor_info(batch[f"A_{adj_count}"], default_value)
            batch_group.create_dataset(f"A_{adj_count}_idx", data=A_data[0])
            batch_group.create_dataset(f"A_{adj_count}_values", data=A_data[1])
            batch_group.create_dataset(f"A_{adj_count}_shape", data=A_data[2])
            adj_count += 1
            
            A_data = get_sparse_tensor_info(batch[f"A_{adj_count}"], default_value)
            batch_group.create_dataset(f"A_{adj_count}_idx", data=A_data[0])
            batch_group.create_dataset(f"A_{adj_count}_values", data=A_data[1])
            batch_group.create_dataset(f"A_{adj_count}_shape", data=A_data[2])
            adj_count += 1
    
    if edges: 
        E_data = get_sparse_tensor_info(batch["E_1"], default_value)
        batch_group.create_dataset("E_1_idx", data=E_data[0])
        batch_group.create_dataset("E_1_values", data=E_data[1])
        batch_group.create_dataset("E_1_shape", data=E_data[2])
        
        E_data = get_sparse_tensor_info(batch["E_2"], default_value)
        batch_group.create_dataset("E_2_idx", data=E_data[0])
        batch_group.create_dataset("E_2_values", data=E_data[1])
        batch_group.create_dataset("E_2_shape", data=E_data[2])
    
    hf.close()
    
def add_graph_to_batch(batch, graph_sample, num_levels, edges=True):
    if len(batch["V_1"]) == 0:
        for key in batch.keys():
            batch[key] = graph_sample[key]
            
    else:
        batch["names"] = np.append(batch["names"], graph_sample["names"], axis=0)
        batch["idx"] = np.append(batch["idx"], graph_sample["idx"], axis=0)
        batch["labels"] = np.append(batch["labels"], graph_sample["labels"], axis=0)
        
        adj_count = 2
    
        for level in range(1, num_levels+1):
            batch[f"V_{level}_keys"] = np.append(batch[f"V_{level}_keys"], graph_sample[f"V_{level}_keys"], axis=0)
            batch[f"V_{level}"] = np.append(batch[f"V_{level}"], graph_sample[f"V_{level}"], axis=0)

            #batch[f"A_{adj_count}"] = disjoint_adj(batch[f"A_{adj_count}"], graph_sample[f"A_{adj_count}"])
            #adj_count += 1
            #if level > 1:

            if level != 1:
                batch[f"A_{adj_count}"] = disjoint_adj(batch[f"A_{adj_count}"], graph_sample[f"A_{adj_count}"])
                adj_count += 1
                
                batch[f"A_{adj_count}"] = disjoint_adj(batch[f"A_{adj_count}"], graph_sample[f"A_{adj_count}"])
                adj_count += 1

                batch[f"A_{adj_count}"] = disjoint_adj(batch[f"A_{adj_count}"], graph_sample[f"A_{adj_count}"])
                adj_count += 1
        if edges:
            batch["E_1"] = disjoint_adj(batch["E_1"], graph_sample["E_1"])
            batch["E_2"] = disjoint_adj(batch["E_2"], graph_sample["E_2"])
        
    return batch