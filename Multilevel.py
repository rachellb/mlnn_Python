import pandas as pd
import NearestNeighborSearch
from sklearn.model_selection import train_test_split
#import MLD
#import neuralNetwork
import time
import numpy as np


def Multilevel(data, dataName, max_ite=1, prop=0.8, multilevel=1, n_neighbors=10,
    Upperlim=500, Imb_size=300, Model_Selec=1, numBorderPoints=10, 
    coarse=0, loss="cross", Level_size=1, refineMethod="border", 
    epochs=100, patience_level=2, weights=False, label=""):
    """
    The main function. Takes data and trains either a multilevel or traditional neural network

    Inputs:
        data: The dataset
        dataName: The name of dataset being used.
        max_ite: Number of iterations to average results over
        prop: Proportion of data to split into test/train  
        n_neighbors: Number of neighbors to use for coarsening
        Upperlim: Maximum size of data in each class for Training
        Imb_size: If the size of each class is less than Imb_size, then perform coarsening
        Model_Selec: Perform hyperparameter optimization (1=Yes, 0=No)
        numBorderPoints: Number of border points to use in refinement
        coarse: Indicator for whether or not at coarsest level
        loss: Loss function to use, "focal" or "cross"
        Level_size: # of levels. For keeping track of depth of coarsening
        refineMethod: Which refinement method to use, "border" or "flip"
        epochs: Number of epochs to train neural network for
        patience_level: Patience for refinement - if no improvement after n levels, stop refining and return best results
        weights: Whether or not to weight the loss function
        label: Label to give results file. 
    """

    data = pd.read_csv(data, index_col=False)
    Pdata = data[data["Label"] == 1]
    Ndata = data[data["Label"] == 2]
    Results = {}
    totalTime = ()



    for ite in range(1, max_ite+1): 
        start = time.time()

        train_lbl, train_data, test_lbl, test_data = Split()

        Pweight = 1/len(Ptraindata)                      
        Nweight = 1/len(Ntraindata) 
        
        traindata = pd.concat(Ntraindata, Ptraindata)
        train_l = pd.concat(Ntrainlbl, PcoarseLbl)


        Best = {} # Will contain best results found. 
        if multilevel == 1:
            # Create the KNN graph that will be used in multilevel learning
            nresult, ndistances, NAD1 = NearestNeighborSearch(Ntraindata, n_neighbors)
            presult, pdistances, PAD1 = NearestNeighborSearch(Ptraindata, n_neighbors)

"""
            Results[ite],posBorderData, negBorderData, Level_size, trainedNetwork, options, Best, flag, Level_results = MLD(traindata, train_l, Best)
            depth = np.mean(Level_size)
        else:
            Results[ite], trainedNetwork[ite], options[ite] = neuralNetwork(Ntestdata, Ptestdata)
            depth = 0

        end = time.time()
        totalTime.append(start-end) 

    averageTime = np.mean(totalTime)     
    # Save results into an excel file
    formatFilename = "Results/%s/Multilevel_%depochs%dRefine%sBorderPoints%dNeighbors%dLoss%s%smaxIte%d.xlsx"
    filename = formatFilename % (dataName, Multilevel, epochs, refineMethod, numBorderPoints, n_neighbors, loss, label, max_ite)

    resultsTable = pd.DataFrame({
        'GMean': Results.GMean, 
        'Acc': Results.Acc, 
        'Sen': Results.Sen,  
        'Spec': Results.Spec, 
        'stdGMean': Results.stdGMean, 
        'stdAcc': Results.stdAcc, 
        'stdSen': Results.stdSen,  
        'stdSpec': Results.stdSpec, 
        'Time (sec)': averageTime, 
        'Depth': depth, 
        'Refined': Results.refined
    }, columns=['GMean','Acc', 'Sen', 'Spec','stdGMean','stdAcc','stdSen', 'stdSpec', 'Time (sec)', 'Depth', 'Refined'])

    resultsTable.to_excel(filename, sheet_name='Sheet1', index=False)

"""


if __name__ == "__main__":
    Multilevel(data="../Hypothyroid.csv", dataName="Hypothyroid")
