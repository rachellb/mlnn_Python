import neuralNetwork
import pandas as pd

def refine(model, Ncoarse, Pcoarse, Nfine, Pfine, options):

    ''' Updates the training data and trains the neural network. Has 2 parts:

    1. Finds the border points in the coarse level using the neural network trained at that level
    2. Finds the nearest neighbors of those points in the fine level data, and adds them together
    to make a new training dataset

    Inputs: 
        <trainedNetwork>: The network that has been trained at the previous level of refinement
        <traindata>: The previous training data
        <train_l>: Labels of previous refinement training data

    Outputs:
        <updatedData>:
    '''
    
    # Find border points
    negBorderIndices = findBorderPoints(model, Ncoarse["Data"], options["numBorderPoints"],
                                                        options["refineMethod"])
    posBorderIndices = findBorderPoints(model, Pcoarse["Data"], options["numBorderPoints"],
                                                        options["refineMethod"])
    # Update training data
    Ntrain = Update_Train_Data(negBorderIndices, Nfine, options["n_neighbors"])
    Ptrain = Update_Train_Data(posBorderIndices, Pfine, options["n_neighbors"])

    traindata = pd.concat([Ntrain["Data"], Ptrain["Data"]])
    train_lbl = pd.concat([Ntrain["Labels"], Ptrain["Labels"]])

    traindata.reset_index(drop=True, inplace=True)
    train_lbl.reset_index(drop=True, inplace=True)

    return traindata, train_lbl

def findBorderPoints(model, traindata, numBorderPoints, refineMethod):
    ''' Finds which points are the "border points", or pseudo-support vectors. 
    Can be done either using the output of the loss function or by using the flip points.
    
    '''
    if refineMethod == 'border':
        """
        This version takes the predictions of the neural network and finds the most "uncertain" ones; 
        those closest to 0.5. 
        """

        # Get predicted values for training data
        predictions = model.predict(traindata)

        # Calculate the distances of each point to the threshold (0.5)
        distances = [abs(0.5-x) for x in predictions]

        # The K smallest distances become our border points
        borderIndices = sorted(range(len(distances)), key=lambda sub: distances[sub])[:numBorderPoints]

    #else:
        # Perform "flip point" method 

        #posBorderPoints,negBorderPoints= flipPointFunction(numBorderPoints)

    return borderIndices


def Update_Train_Data(borderIndices, fineData, n_neighbors):
    ''' Takes the border points and finds their nearest neighbors in the higher, "fine" level of data.
    These are combined with the border points to make our new dataset .

    Inputs:
        <posBorderData>:
        <negBorderData>:
        <NfineData>:
        <NfineLbl>:
        <>
    Outputs:
        <>
    '''

    # Adds in the border points
    train = {}

    borderData = fineData["Data"].iloc[borderIndices, :]
    borderLabels = fineData["Labels"][borderIndices]

    # Need to add in the nearest neighbors of the border points
    neighborsInd = fineData["KNeighbors"][borderIndices, :]
    neighborData = fineData["Data"].iloc[neighborsInd.flatten()]
    neighborLbl = fineData["Labels"][neighborsInd.flatten()]

    train["Data"] = pd.concat([borderData, neighborData])
    train["Labels"] = pd.concat([borderLabels, neighborLbl])

    #train["Data"] = train["Data"].drop_duplicates()
    train["Data"] = train["Data"][~train["Data"].index.duplicated(keep='first')]
    train["Labels"] = train["Labels"][~train["Labels"].index.duplicated(keep='first')]

    return train
