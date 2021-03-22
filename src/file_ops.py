import os
import random
import numpy as np
import math


def load_brep_features(base_path):
    brep_keys = []
    features = []
    labels = []
    
    with open(base_path + "_facefeature.csv", 'r') as file:
        for i, line in enumerate(file):
            if i == 0:  # Skip first line (header)
                continue
            s = line[:-1].split(',')
            brep_keys.append(s[0])
            features.append(list(map(float, s[1:6])))
            labels.append(s[6])
    
    features = np.array(features, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)
    
    return brep_keys, features, labels


def load_brep_adj(base_path, brep_keys):
    brep_adj = np.zeros((len(brep_keys), len(brep_keys)))
    convex_adj = np.zeros((len(brep_keys), len(brep_keys)))
    concave_adj = np.zeros((len(brep_keys), len(brep_keys)))
    other_adj = np.zeros((len(brep_keys), len(brep_keys)))
    
    with open(base_path + "_faceadj.csv", 'r') as file:
        for i, line in enumerate(file):
            if i == 0:  # Skip first line (header)
                continue
            s = line[:-1].split(',')
            a = brep_keys.index(s[0])
            if s[1] == "Null": 
                continue
            b = brep_keys.index(s[1])
            brep_adj[a, b] = 1
            brep_adj[b, a] = 1
            
            if s[2] == "convex_c":
                convex_adj[a, b] = 1
                convex_adj[b, a] = 1
            elif s[2] == "concave_c":
                concave_adj[a, b] = 1
                concave_adj[b, a] = 1 
            elif s[2] == "smooth_ccv_c":
                concave_adj[a, b] = 1
                concave_adj[b, a] = 1 
            elif s[2] == "smooth_cvx_c":
                convex_adj[a, b] = 1
                convex_adj[b, a] = 1
            elif s[2] == "smooth_flat_c":
                other_adj[a, b] = 1
                other_adj[b, a] = 1  
            elif s[2] == "smooth_var_c":
                other_adj[a, b] = 1
                other_adj[b, a] = 1            
            elif s[2] == "smooth_inf_c":
                other_adj[a, b] = 1
                other_adj[b, a] = 1             
            elif s[2] == "knife_cvx_c":
                convex_adj[a, b] = 1
                convex_adj[b, a] = 1          
            elif s[2] == "knife_ccv_c":
                concave_adj[a, b] = 1
                concave_adj[b, a] = 1           
            elif s[2] == "variable_c":
                other_adj[a, b] = 1
                other_adj[b, a] = 1      

    return brep_adj, convex_adj, concave_adj, other_adj


def load_facet_features(base_path):
    facet_keys = []
    features = []
    
    with open(base_path + "_facetfeature.csv") as file:
        for i, line in enumerate(file):
            if i == 0:  # Skip first line (header)
                continue
            s = line[:-1].split(',')
            facet_keys.append(s[0])
            features.append(list(map(float, s[1:])))
            
    features = np.array(features, dtype=np.float32)
    
    return facet_keys, features


def load_facet_adj(base_path, facet_keys):
    facet_adj = np.zeros((len(facet_keys), len(facet_keys)))
    
    with open(base_path + "_facetadj.csv", 'r') as file:
        for i, line in enumerate(file):
            if i == 0:  # Skip first line (header)
                continue
            s = line[:-1].split(',')
            a = facet_keys.index(s[0])
            b = facet_keys.index(s[1])
            facet_adj[a, b] = 1
            facet_adj[b, a] = 1
            
    return facet_adj


def load_face_facet_link(base_path, brep_keys, facet_keys):
    projection = np.zeros((len(brep_keys), len(facet_keys)))
    with open(base_path + "_facefacetlink.csv", 'r') as file:
        
        for i, line in enumerate(file):
            if i == 0:  # Skip first line (header)
                continue
            s = line[:-1].split(',')
            a = brep_keys.index(s[0])
            b = facet_keys.index(s[1])
            projection[a, b] = 1
            
    embedding = np.transpose(projection)
    
    return embedding, projection


def get_split(directory, split):
    sub_dirs = [f.path for f in os.scandir(directory) if f.is_dir()]
    random.shuffle(sub_dirs)
    random.shuffle(sub_dirs)
    num_dirs = len(sub_dirs)
    
    train_split = int(math.ceil(num_dirs * split["train"]))
    val_split = int(math.ceil(num_dirs * split["val"]))
    
    train_dirs = sub_dirs[:train_split]
    val_dirs = sub_dirs[train_split:train_split+val_split]
    test_dirs = sub_dirs[train_split+val_split:]
    
    return train_dirs, val_dirs, test_dirs
