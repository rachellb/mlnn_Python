import numpy as np
from pynndescent  import NNDescent
import hnswlib

def NearestNeighborSearch(data, n_neighbors=10, metric='euclidean', method="hnsw"):

    """
    Inputs:
            data: Dataframe that you want to find nearest neighbors of
            n_neighbors: How many nearest neighbors you want to find
            metric: The distance metric you're using for nearest neighbor search

       Outputs: 
            neighbors: Matrix of indices for nearest neighbors. 
               column: The point in question
               row: The index of the given point's i nearest neighbor.
            distances: The distances between each point and its nearest neighbor
            IndicatorMatrix: A matrix indicating which points are nearest neighbors. If two 
                               points are nearest neighbors, they are indicated with a 1 in
                               the corresponding point in the matrix.  
    """

    if method == "nndescent":
        dataMatrix = data.values
        index = NNDescent(dataMatrix, metric=metric)
        neighbors = index.neighbor_graph[0][:, 1:(n_neighbors+1)] # Select only the k nearest neighbors

    else:
        dim = data.shape[1]
        num_elements = data.shape[0]
        ids = np.arange(num_elements)

        # Declaring index
        p = hnswlib.Index(space='l2', dim=dim)  # possible options are l2, cosine or ip

        # Initializing index - the maximum number of elements should be known beforehand
        p.init_index(max_elements=num_elements, ef_construction=200, M=16)

        # Element insertion (can be called several times):
        p.add_items(data, ids)

        # Controlling the recall by setting ef:
        p.set_ef(50)  # ef should always be > k

        # Query dataset, k - number of the closest elements (returns 2 numpy arrays)
        neighbors, distances = p.knn_query(data, k=n_neighbors)

    return neighbors
