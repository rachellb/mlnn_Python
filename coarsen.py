import numpy as np
import NearestNeighborSearch

def coarsen(IndicatorMatrix, data, lbl, n_neighbors=10, metric='euclidean', T=0.6):
    ''' The primary coarsening function. Uses Maximum Independent Set to coarsen.
    Inputs: 
        <IndicatorMatrix>: A matrix indicating which points are nearest neighbors. If two 
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
        <IndicatorMatrix>: A matrix indicating which points are nearest neighbors in coarsened dataset. If two 
                               points are nearest neighbors, they are indicated with a 1 in
                               the corresponding point in the matrix, otherwise 0. 
    '''
    
    R = findCoarseIndicies(IndicatorMatrix, T)
    l = len(data, 1)
    node = list(range(1,l+1)) # indices of all points in fine data
    Vhat = R.sort() # The sorted list of caorsened nodes
    Comp_Vhat = np.setxor1d(node,Vhat) # Return a list of all points that are not in their intersection
    data = data[Comp_Vhat] # Remove datapoints that are not in the coarsened set 
    lbl = lbl[Comp_Vhat] # Remove labels that are not in the coarsened set 

    # Calculate the distance and indices of the ten nearest neighbors, as well as their corresponding
    # Indicator matrix 
    result, distances, IndicatorMatrix = NearestNeighborSearch(data, n_neighbors, metric)

    return data, lbl, result, distances, IndicatorMatrix


def findCoarseIndicies(IndicatorMatrix, T):

    '''
    Inputs: 
        <IndicatorMatrix>: A matrix indicating which points are nearest neighbors. If two 
                               points are nearest neighbors, they are indicated with a 1 in
                               the corresponding point in the matrix, otherwise 0. 
        <T>: Proportional size of coarse data relative to fine data. 
    Outputs: 
        <R>: The Maximum Independent Set. 
        Contains a list of indicies of points that remain in the coarse set. 
    '''
    T = 0.6 # Proportion of graph that will remain in the "coarse" level
    n, m = IndicatorMatrix.shape # Size of 
    l = np.arange(n)

    R = []
    while len(R) < T * n:
        l = np.arange(n)
        l[R] = 0
        while np.count_nonzero(l) > 0:
            toPick = np.where(l)[0]
            i = np.random.choice(toPick)
            R.append(i)
            l[i] = 0
            neigh = np.where(IndicatorMatrix[i,:])[0]
            l[neigh] = 0
    
    return R