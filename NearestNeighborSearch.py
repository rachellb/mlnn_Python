import numpy as np
from PyNNDescent import NNDescent

def NearestNeighborSearch(data, n_neighbors=10, metric='euclidean'):
    """
    Inputs:
            data: Dataframe that you want to find nearest neighbors of
            n_neighbors: How many nearest neighbors you want to find
            metric: The distance metric you're using for nearest neighbor search

       Outputs: 
            result: Matrix of indices for nearest neighbors. 
               column: The point in question
               row: The index of the given point's i nearest neighbor.
            distances: The distances between each point and its nearest neighbor
            IndicatorMatrix: A matrix indicating which points are nearest neighbors. If two 
                               points are nearest neighbors, they are indicated with a 1 in
                               the corresponding point in the matrix.  
    """

    dataMatrix = data.values
    nnd = NNDescent(dataMatrix, metric=metric)
    nnd.init_network()
    nnd.search(k=n_neighbors)
    result = nnd.get_knn()
    distances = nnd.get_distances()
    IndicatorMatrix = np.zeros((dataMatrix.shape[0], dataMatrix.shape[0]))
    for point in range(dataMatrix.shape[0]):
        for neighbor in range(result[point].shape[0]):
            IndicatorMatrix[result[point][neighbor]][point] = 1

    return result, distances, IndicatorMatrix
