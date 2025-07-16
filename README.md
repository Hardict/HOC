# HOC

The file HOC.py defines the construction method of hierarchical overlapping clustering graphs and a naive approach for computing costs.

The file fast_algorithm_for_2OC.py provides the specific implementation of the 2-OC algorithm, with its direct interface being the greedy_select_overlap_greedy function.

The file KOC_v2.py contains the implementation of the k-HOC algorithm, including custom graph generation and the find_k_clusters function that repeatedly calls 2-OC to solve the k-HOC problem. Additionally, it includes experimental code related to the MNIST dataset in KOC_mnist.py.
