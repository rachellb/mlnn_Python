import neuralNetwork

def refine(refineMethod, trainedNetwork, traindata, train_l, NfineData, NfineLbl,PfineData,PfineLbl,n_neighbors,nresultFine,presultFine,numBorderPoints):
    ''' Updates the training data and trains the neural network. Has 3 parts: 
    1. Finds the current border points
    2. Updates the training data
    3. Train the neural network on updated data 

    Inputs: 
        <trainedNetwork>: The network that has been trained at the previous level of refinement
        <traindata>: The previous training data
        <train_l>: Labels of previous refinement training data

    Outputs: 
        <Results>:
        <trainedNetwork>: 

    '''
    
    # Find border points
    posBorderPoints, negBorderPoints = findBorderPoints(trainedNetwork, traindata, train_l, numBorderPoints, refineMethod)

    # Update training data
    traindata,train_l = Update_Train_Data(posBorderPoints, negBorderPoints,NfineData,NfineLbl,PfineData,PfineLbl,n_neighbors,nresultFine,presultFine)
     
    # TODO: Double check status of validation data
    # Train on updated dataset
    Results, trainedNetwork, options = neuralNetwork(traindata,train_l,valdata,val_l,loss,
                                                     epochs,weights,trainedNetwork, options,Model_Selec=0, model=None)
       
    return Results, trainedNetwork

def findBorderPoints(trainedNetwork, traindata, train_l, numBorderPoints, refineMethod):
    ''' Finds which points are the "border points", or pseudo-support vectors. 
    Can be done either using the output of the loss function or by 
    
    '''
    if refineMethod == '':
        # TODO: Fix this
        posBorderPoints, negBorderPoints = numBorderPoints

    else: 
        # do other stuff. 

        posBorderPoints,negBorderPoints= flipPointFunction(numBorderPoints)


    return posBorderPoints, negBorderPoints


def Update_Train_Data(posBorderData, negBorderData,NfineData,NfineLbl,PfineData,PfineLbl,n_neighbors,nresultFine,presultFine):
    ''' Takes the border points and finds their nearest neighbors in the higher, "fine" level of data.
    These are combined with the border points to make our new dataset .

    Inputs:
        <>

    Outputs:
        <>
    '''
    
    #finding support vectors in the nresult
    [~,nindx] = ismember(negBorderData.X,Ndata,'rows');

    return