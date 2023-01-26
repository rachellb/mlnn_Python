import pandas as pd
import coarsen
import neuralNetwork
import refine

def MLD(traindata, train_l, testdata, test_l, NfineData, NfineLbl, NcoarseData, NcoarseLbl, PfineData, PfineLbl, PcoarseData, PcoarseLbl, PAD, Upperlim, Pweight, Nweight,n_neighbors,level,nresult1,presult1,nresult,presult,U_trainsize,Model_Selec,Imb_size,coarse,epochs, Multilevel,MoS_UB,Level_size, numBorderPoints,loss, refineMethod, patience_level, weights, Best):
    ''' A recursive function that iteratively coarsens the data, 
    trains the network once we hit the coarsest level, then begins refinement.
       Inputs:
            <traindata>: Training Data, both positive and negative
            <train_l>: Training Labels
            <testdata>: Test data, both positive and negative
            <test_l>: Testing Labels
            <NfineData>: Negative data for previous refinement level
            <NfineLbl>: Negative labels for previous refinement level
            <NcoarseData>: Negative data for this refinement level
            <NcoarseLbl>: Negative labels for this refinement level
            <NAD>: Nearest Neighbor Indicator Matrix for Negative Points
            <PfineData>: Positive data for previous refinement level
            <PfineLbl>: Positive labels for previous refinement level
            <PcoarseData>: Positive data for this refinement level
            <PcoarseLbl>: Positive labels for this refinement level
            <PAD>: Nearest Neighbor Indicator Matrix for Positive Points
            <Upperlim>: Maximum size of coarsest level data
            <Pweight>: Weights for positive data, if desired
            <Nweight>: Weights for negative data, if desired
            <n_neighbors>: How many nearest neighbors to select
            <level>: Which level of refinement we are at
            <nresult1>: Nearest neighbors of negative data at this refinement level
            <presult1>: Nearest neighbors of positive data at this refinement level
            <nresult>: Nearest neighbors of negative data at previous refinement level
            <presult>: Nearest neighbors of positive data at previous refinement level
            <U_trainsize>:
            <Model_Selec>:
            <Imb_size>: Maximum size of positive and negative data individually 
            <coarse>: Binary variable indicating if we have hit the coarsest level. 
            <epochs>: Number of epochs to train neural network for
            <Multilevel>: Whether or not we're doing the multilevel version of the code. 
            <Level_size>: What level we're at
            <numBorderPoints>: How many border points (for each class) to select during refinement
            <loss>: Which type of loss function to use
            <refineMethod>: Which refinement method to use, "flip" or "border" 
            <patience_level>: How many levels to tolerate no improvement before stopping refinement
            <weights>: Neural Network weights
            <Best>: Dictionary of best results found so far
       Outputs: 
           <Results>: The results of the model at this level
           <posBorderData>: Positive Border points 
           <negBorderData>: Negative Border points
           <Level_size>: 
           <trainedNetwork>: The trained neural network for this level
           <options>: Neural network hyperparameters
           <Best>: Dictionary of best results found so far
           <flag>: Flag for ceasing refinement if results have not improved in given number of levels
           <Level_results>: Results per refinement level
   '''
    
    DATA_size = len(traindata,1)

    # If the combined positive and negative data is below the maximum training threshold, 
    # begin training. 
    if DATA_size < Upperlim | coarse == 1:
        Level_size = level+1

        # TODO: Make sure you've separated the validation data from the test data!
        Results, trainedNetwork, options, posBorderData, negBorderData = neuralNetwork(traindata,train_l,testdata,test_l,loss,epochs,weights,Multilevel, numBorderPoints, refineMethod)
        coarse = 0 # Training of coarsest section is done.

        # Results of best trained neural network so far. 
        Best["level"] = level+1
        Best["GMean"] = Results["GMean"]
        Best["Acc"] = Results["Acc"]
        Best["Sen"] = Results["Sen"]
        Best["Spec"] = Results["Spec"]

    # Else, begin coarsening the data
    else:
        # Keep track of information regarding previous refinement levels
        # Previous levels are considered "fine" relative to the current level. 
        NfineData = NcoarseData
        NfineLbl = NcoarseLbl
        PfineData = PcoarseData
        PfineLbl = PcoarseLbl
        nresultFine = nresultCoarse 
        presultFine = presultCoarse 

        # If there are too many points in each class to begin training, begin coarsening
        if (len(NfineData,1) > Imb_size):
            NcoarseData, NcoarseLbl, nresultCoarse, ndistancesCoarse, NAD = coarsen(NAD, NfineData, NfineLbl, n_neighbors, T=0.6)

        if (len(PfineData, 1) > Imb_size):
            PcoarseData, PcoarseLbl, presultCoarse, pdistancesCoarse, PAD = coarsen(PAD, PfineData, PfineLbl, n_neighbors, T=0.6)

        traindata = pd.concat(NcoarseData, PcoarseLbl)
        train_l = pd.concat(NcoarseLbl, PcoarseLbl)
        Pweight = 1/len(PcoarseLbl,1) 
        Nweight = 1/len(NcoarseLbl,1) 

        # If the size of each dataset is considered small enough or no more meaningful coarsening
        # can be performed, then this is the coarsest level of data. 
        if ((len(NcoarseData) < Imb_size) & (len(PcoarseData) < Imb_size)) | (len(NcoarseData)==len(NfineData)):
            coarse = 1

        # Go to next iteration of recursion
        Results,posBorderData, negBorderData, Level_size, trainedNetwork, options, Best, flag, Level_results = MLD(traindata, 
        train_l, testdata, test_l, NfineData, NfineLbl, NcoarseData, NcoarseLbl, PfineData, PfineLbl, PcoarseData, PcoarseLbl, 
        PAD, Upperlim, Pweight, Nweight,n_neighbors,level,nresult1,presult1,nresult,presult,U_trainsize,Model_Selec,Imb_size,coarse,
        epochs, Multilevel,MoS_UB,Level_size, numBorderPoints,loss, refineMethod, patience_level, weights, Best)

        # Once all of the coarsening has been performed, begin refining the model
        traindata,train_l,Results = refine(trainedNetwork, traindata, train_l, numBorderPoints)

        #Check if current refinement gives best results
        if Results.GMean > Best.GMean:
            Best.GMean = Results.GMean
            Best.Acc = Results.GMean
            Best.Sen = Results.GMean
            Best.Spec = Results.Spec
            Best.level = level
            Best.difference = Level_size-level; # How much did we refine? 
        

        #If best was beyond patience level, stop refinement
        if (Best.level - level) >= patience_level: 
            flag = 1

    return Results,posBorderData, negBorderData, Level_size, trainedNetwork, options, Best, flag, Level_results