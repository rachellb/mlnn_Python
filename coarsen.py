import numpy as np
import NearestNeighborSearch

def coarsen(AdjMatrix, data, lbl, n_neighbors=10, metric='euclidean', T=0.6):
    ''' The primary coarsening function. Uses Maximum Independent Set to coarsen.
    Inputs: 
        <AdjMatrix>: A matrix indicating which points are nearest neighbors. If two
                               points are nearest neighbors, they are indicated with a 1 in
                               the corresponding point in the matrix, otherwise 0. 
        <data>: The fine data to be coarsened
        <lbl>: Corresponding labels to fine data
        <n_neighbors>: Number of points to consider in nearest neighbor search
        <metric>: The distance metric to be used in nearest neighbor search
        <T>: Proportional size of coarse data relative to fine data. 
    Outputs: 
        <data>: The coarsened data
        <lbl>: The coarsened data's corresponding labels
        <result>: Matrix of indices for nearest neighbors. 
               column: The point in question
               row: The index of the given point's i nearest neighbor.
        <distances>: The distances between each point and its nearest neighbor
        <AdjMatrix>: A matrix indicating which points are nearest neighbors in coarsened dataset. If two
                               points are nearest neighbors, they are indicated with a 1 in
                               the corresponding point in the matrix, otherwise 0. 
    '''

    # Calculate which points will be in coarsened dataset
    coarseData = DomSetCoarsening(AdjMatrix, T)


    l = len(data)
    node = list(range(1,l+1)) # indices of all points in fine data
    vhat = coarseData.sort() # The sorted list of coarsened nodes
    comp_Vhat = np.setxor1d(node, vhat) # Return a list of all points that are not in their intersection

    data = data[comp_Vhat] # Remove datapoints that are not in the coarsened set
    lbl = lbl[comp_Vhat] # Remove labels that are not in the coarsened set

    # Calculate the distance and indices of the ten nearest neighbors, as well as their corresponding
    # Indicator matrix 
    result, AdjMatrixCoarse = NearestNeighborSearch(data, n_neighbors, metric)

    return data, lbl, result, AdjMatrixCoarse


def DomSetCoarsening(AdjMatrix, T=0.6):

    '''
    This function takes in the adjacency matrix and creates a maximum independent set.
    It's a greedy algorithm that finds a series of independent sets, adding them
    to the "coarse" dataset. This does not guarantee that we'll have a maximum
    independent set, but the resulting "coarse" data is a dominant set of the fine data.

    Inputs: 
        <IndicatorMatrix>: A matrix indicating which points are nearest neighbors. If two 
                               points are nearest neighbors, they are indicated with a 1 in
                               the corresponding point in the matrix, otherwise 0. 
        <T>: Proportional size of coarse data relative to fine data. 
    Outputs: 
        <maxSet>: The Maximum Independent Set (MIS).
        Contains a list of indicies of points that remain in the coarse set. 
    '''

    n, m = AdjMatrix.shape # Size of indicator matrix.
    domSet = []

    # Until we have a desired fraction (T) of the fine data, repeat this loop
    while len(domSet) < T * n:
        coarseOptions = np.arange(n) # Create a list of possible choices
        coarseOptions[domSet] = 0 # Remove our previous independent set from the option list
        while np.count_nonzero(coarseOptions) > 0:
            toPick = np.where(coarseOptions)[0] # Select all non-zero options from l
            i = np.random.choice(toPick) # Pick one of those at random
            domSet.append(i) # add it to the independent set list
            coarseOptions[i] = 0 # remove it from the list of options
            neigh = np.where(AdjMatrix[i,:])[0] # Find the neighors of that random point
            coarseOptions[neigh] = 0 # Remove all neighbors from the options list as well
    
    return domSet