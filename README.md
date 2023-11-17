# Hierarchical-CADNet
This repo provides a code of the neural network described in the paper:
**Hierarchical CADNet: Learning from B-Reps for Machining Feature Recognition**.

It is a deep learning approach to learn machining features from CAD models. To achieve this, the geometry of the CAD models are described by discretising the surface of the CAD model into a mesh. This mesh can then be treated as a graph and operated on by a graph neural network. The overall topology of the CAD model can be described by the face adjacency matrix. A hierarchical graph structure can be constructed by between the B-Rep adjacency graph and the mesh. A STL mesh was chosen as the tessellation method due to its wide availability in CAD system and offers a concise representation. Each facet in the mesh denotes a vertex in a level of the hierarchical graph. Each of these vertices contain information of the facet’s planar equation, used to describe the surface. A second level of the hierarchical graph denotes the B-Rep adjacency graph. There exists persistent links between each B-Rep face vertex and their corresponding STL facet vertex. A B-Rep face vertex can have more than one STL facets adjacent to it. The goal of the approach is to be able to classify the machining feature of each B-Rep face vertex in the graph.

![](imgs/hierarchical_graph_structure.png)

## Requirements
- Python >= 3.8.5
- Tensorflow >= 2.2.0
- h5py >= 1.10.6
- Numpy >= 1.19.1
- Scikit-learn >= 0.23.2

## Instructions
- Generate hierarchical B-Rep graphs and batches using code in this repo: https://github.com/wadaniel/hierarchical-brep-graphs/tree/main
- Place hdf5 dataset files in */data* folder.
- To train Hierarchical CADNet (Edge) which uses edge convexity information run **train_edge.py**, set data type, dataloader file locations.
- To train Hierarchical CADNet (Adj) which uses only adjacency information run **train_adj.py**, set data type, dataloader file locations.
- To train Hierarchical CADNet (Single) which is a graph classification task run **train_single_feat.py**, set data type, dataloader file locations.

## Visualization
There is a basic CAD viewer provided. To use it additional Python packages are required (PythonOCC):
- pythonocc-core >= 7.4.1 (more info here: https://github.com/tpaviot/pythonocc-core)
- occt >= 7.4.0 (more info here: https://github.com/tpaviot/pythonocc-core)

To test a single CAD model with a trained network model and save a STEP file with the predicted labels, the `test_and_save.py` script can be used.
A directory of STEP files can be viewed using the `visualizer.py` script, in which each label has a unique color.

## Test Cases
In the Hierarchical CADNet paper, Section 6.4 discussed results on more complex test cases. The STEP files for these CAD models can be found in the `test_cases` directory. These are labelled in the same way as the MFCAD++ dataset, with a label id being attributed to each B-Rep face in the CAD model.

## Citation
Please cite this work if used in your research:

    @article{hierarchicalcadnet2022,
      Author = {Andrew R. Colligan, Trevor. T. Robinson, Declan C. Nolan, Yang Hua, Weijuan Cao},
      Journal = {Computer-Aided Design},
      Title = {Hierarchical CADNet: Learning from B-Reps for Machining Feature Recognition},
      Year = {2022}
      Volume = {147}
      URL = {https://www.sciencedirect.com/science/article/abs/pii/S0010448522000240}
    }

## Funding
This project was funded through DfE funding.
