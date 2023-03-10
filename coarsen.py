import numpy as np
from NearestNeighborSearch import NearestNeighborSearch

import time

def coarsen(fineData, n_neighbors=10, metric='euclidean', T=0.6):
    ''' The primary coarsening function. Uses dominant set to coarsen currently.
    Inputs:
        <fineData>:
            <AdjMatrix>: A matrix indicating which points are nearest neighbors. If two
                                   points are nearest neighbors, they are indicated with a 1 in
                                   the corresponding point in the matrix, otherwise 0.
            <Data>: The fine data to be coarsened
            <Label>: Corresponding labels to fine data
            <KNeighbors>: Matrix of indices for nearest neighbors.
                   column: The point in question
                   row: The index of the given point's k nearest neighbor.
        <n_neighbors>: Number of points to consider in nearest neighbor search
        <metric>: The distance metric to be used in nearest neighbor search
        <T>: Proportional size of coarse data relative to fine data.

    Outputs:
        <coarseData>: A dictionary containing information about the Coarse dataset.
            <Data>: The coarsened data
            <Label>: The coarsened data's corresponding labels
            <KNeighbors>: Matrix of indices for nearest neighbors.
                   column: The point in question
                   row: The index of the given point's i nearest neighbor.
            <AdjMatrix>: A matrix indicating which points are nearest neighbors in coarsened dataset. If two
                                   points are nearest neighbors, they are indicated with a 1 in
                                   the corresponding point in the matrix, otherwise 0.
    '''

    coarseData = {}

    # Calculate which points will be in coarsened dataset
    coarseIndicies = iterativeIndependentSet(fineData, T)

    coarseData["Data"] = fineData["Data"].iloc[coarseIndicies, :]
    coarseData["Labels"] = fineData["Labels"][coarseIndicies]

    coarseData["Data"].reset_index(drop=True, inplace=True)
    coarseData["Labels"].reset_index(drop=True, inplace=True)

    # Calculate nearest neighbors of this new coarse dataset.
    #coarseData["KNeighbors"], coarseData["AdjMatrix"] = NearestNeighborSearch(coarseData["Data"], n_neighbors, metric)

    coarseData["KNeighbors"], coarseData["Weights"] = NearestNeighborSearch(coarseData["Data"], n_neighbors, metric)

    return coarseData

def iterativeIndependentSet(fineData, T=0.6):

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

    n, m = fineData["KNeighbors"].shape # Size of indicator matrix.
    domSet = []

    # Until we have a desired fraction (T) of the fine data, repeat this loop
    while len(domSet) < T * n:
        coarseOptions = np.arange(n) # Create a list of possible choices
        coarseOptions[domSet] = 0 # Remove our previous independent set from the option list

        while np.count_nonzero(coarseOptions) > 0:
            startInner = time.time()
            toPick = np.where(coarseOptions)[0] # Select all non-zero options from l
            i = np.random.choice(toPick) # Pick one of those at random
            domSet.append(i) # add it to the independent set list
            coarseOptions[i] = 0 # remove it from the list of options
            neigh = fineData["KNeighbors"][i,:]
            coarseOptions[neigh] = 0 # Remove all neighbors from the options list as well
            endInner = time.time()
            total = endInner - startInner

    return domSet



def algebraicMultigrid(fineData, eta=2, Q=0.6):

    Seeds = [] # Starts as empty list

    # Sum weights in advance (faster than doing in the loop)

    # numpy version
    fineData["sumWeights"] = np.sum(fineData["Weights"], axis=1)

    # Step 1: Calculate future volumes for all point in the fine data set
    if "volume" not in fineData:
        # If this is the finest set, set volumes at this level to 1
        fineData["volume"] = np.ones(fineData["Data"].shape[0])

    # Calculate the future volume ----------------------------------------

    for point in range(fineData["Data"].shape[0]):
        for neighbor in range(fineData["KNeighbors"][point].shape[0]):
            tempVolume = fineData["volume"][neighbor] * (fineData["Weights"][point][neighbor] / fineData["sumWeights"][
                fineData["KNeighbors"][point][neighbor]])
            fineData["futureVolume"][point] = fineData["futureVolume"][point] + tempVolume

            # Multiply result by this point's volume
        fineData["futureVolume"][point] = fineData["futureVolume"][point] * fineData["volume"][point]

    # ---------------------------------------------------------------------
    # Step 2: Create seed set
    # 2.1 - Find Dominant seeds
    averageVolumes = np.sum(fineData["futureVolume"], axis=0)
    averageVolumes = eta*averageVolumes
    dominant = np.where(fineData["futureVolume"] > averageVolumes)[0]
    Seeds.extend(dominant)

    # Then need to remove these from the "fine" dataset.

    # 2.2 - secondary seeds
    sortVolumeIndices = np.argsort(-fineData["futureVolume"])
