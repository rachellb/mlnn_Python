import neuralNetwork

def refine(model, Ncoarse, Pcoarse, n_neighbors, options):

    ''' Updates the training data and trains the neural network. Has 2 parts:

    1. Finds the current border points given the current neural network
    2. Updates the training data with relevant fine level data

    Inputs: 
        <trainedNetwork>: The network that has been trained at the previous level of refinement
        <traindata>: The previous training data
        <train_l>: Labels of previous refinement training data

    Outputs:
        <updatedData>:
    '''
    
    # Find border points
    negBorderPoints = findBorderPoints(model, Ncoarse, options["numBorderPoints"],
                                                        options["refineMethod"])
    posBorderPoints = findBorderPoints(model, Pcoarse, options["numBorderPoints"],
                                                        options["refineMethod"])
    # Update training data
    Nfine = Update_Train_Data(negBorderPoints, Ncoarse, options["n_neighbors"])
    Pfine = Update_Train_Data(posBorderPoints, Ncoarse, options["n_neighbors"])

    return Nfine, Pfine

def findBorderPoints(model, traindata, train_l, numBorderPoints, refineMethod):
    ''' Finds which points are the "border points", or pseudo-support vectors. 
    Can be done either using the output of the loss function or by using the flip points.
    
    '''
    if refineMethod == 'border':
        """
        This should take the output of the neural network
        """
        # TODO: Fix this
        model.predict(traindata)
        posBorderPoints, negBorderPoints = numBorderPoints

    #else:
        # Perform "flip point" method 

        #posBorderPoints,negBorderPoints= flipPointFunction(numBorderPoints)

    return posBorderPoints, negBorderPoints


def Update_Train_Data(borderPoints, coarseData, n_neighbors):
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
    
    #finding support vectors in the nresult
    #~,nindx = ismember(negBorderData.X,NfineData,'rows')

    fineData = "placeholder"

    return fineData