import neuralNetwork

def refine(trainedNetwork, traindata, train_l, NfineData, NfineLbl,PfineData,PfineLbl,n_neighbors, options):

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
    posBorderPoints, negBorderPoints = findBorderPoints(trainedNetwork, traindata, train_l,
                                                        options["numBorderPoints"], options["refineMethod"])

    # Update training data
    traindata,train_l = Update_Train_Data(posBorderPoints, negBorderPoints,NfineData,NfineLbl,PfineData,PfineLbl,n_neighbors)


    return traindata, train_l

def findBorderPoints(trainedNetwork, traindata, train_l, numBorderPoints, refineMethod):
    ''' Finds which points are the "border points", or pseudo-support vectors. 
    Can be done either using the output of the loss function or by 
    
    '''
    if refineMethod == '':
        # TODO: Fix this
        posBorderPoints, negBorderPoints = numBorderPoints

    else: 
        # Perform "flip point" method 

        posBorderPoints,negBorderPoints= flipPointFunction(numBorderPoints)

    return posBorderPoints, negBorderPoints


def Update_Train_Data(posBorderData, negBorderData,NfineData,NfineLbl,PfineData,PfineLbl,n_neighbors,nresultFine,presultFine):
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

    return