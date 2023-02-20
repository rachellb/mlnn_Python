import neuralNetwork
import pandas as pd

def refine(model, Ncoarse, Pcoarse, Nfine, Pfine, n_neighbors, options):

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
    negBorderIndices = findBorderPoints(model, Ncoarse, options["numBorderPoints"],
                                                        options["refineMethod"])
    posBorderIndices = findBorderPoints(model, Pcoarse, options["numBorderPoints"],
                                                        options["refineMethod"])
    # Update training data
    Ntrain = Update_Train_Data(negBorderIndices, Nfine, options["n_neighbors"])
    Ptrain = Update_Train_Data(posBorderIndices, Nfine, options["n_neighbors"])

    traindata = pd.concat([Ntrain["Data"], Ptrain["Data"]])
    train_lbl = pd.concat([Ntrain["Labels"], Ptrain["Labels"]])

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
        distances = abs(0.5-predictions)

        # The K smallest distances become our border points
        borderIndices = sorted(range(len(distances)), key=lambda sub: traindata[sub])[:numBorderPoints]
        #borderPoints = traindata[borderIndices, :]

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

    train = {}
    train["Data"] = fineData["Data"][borderIndices, :]
    train["Labels"] = fineData["Labels"][borderIndices, :]

    return train