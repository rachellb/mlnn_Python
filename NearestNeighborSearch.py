import numpy as np
from pynndescent  import NNDescent

def NearestNeighborSearch(data, n_neighbors=10, metric='euclidean'):

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

    dataMatrix = data.values
    index = NNDescent(dataMatrix, metric=metric)
    neighbors = index.neighbor_graph[0][:, 1:(n_neighbors+1)] # Select only the k nearest neighbors

    """
    # Create a new indicator matrix for this coarse level
    AdjMatrix = np.zeros((dataMatrix.shape[0], dataMatrix.shape[0]))
    for point in range(dataMatrix.shape[0]):
        # For each neighbor of the given point
        for neighbor in range(neighbors[point].shape[0]):
            # Indicate that they are neighbors in the matrix
            AdjMatrix[neighbors[point, neighbor], point] = 1

    """

    return neighbors #, AdjMatrix
