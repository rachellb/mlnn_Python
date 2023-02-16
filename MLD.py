import pandas as pd
from coarsen import coarsen
import neuralNetwork
#import refine
from Evaluate import Evaluate

def MLD(traindata, train_l, valdata, val_l, level, NdataCoarse, PdataCoarse, options,
        NdataFine=None, PdataFine=None, coarse=0, Depth=0, Best=None):

    ''' A recursive function that iteratively coarsens the data,
    trains the network once we hit the coarsest level, then begins refinement.
       Inputs:
            <traindata>: Training Data, both positive and negative
            <train_l>: Training Labels
            <testdata>: Test data, both positive and negative
            <test_l>: Testing Labels
            <NdataFine>:
                <NfineData>: Negative data for previous refinement level
                <NfineLbl>: Negative labels for previous refinement level
                <NcoarseData>: Negative data for this refinement level
                <NcoarseLbl>: Negative labels for this refinement level
                <NAD>: Nearest Neighbor Indicator Matrix for Negative Points
            <PdataFine>:
                <PfineData>: Positive data for previous refinement level
                <PfineLbl>: Positive labels for previous refinement level
                <PcoarseData>: Positive data for this refinement level
                <PcoarseLbl>: Positive labels for this refinement level
                <PAD>: Nearest Neighbor Adjacency Matrix for Positive Points
            <Upperlim>: Maximum size of coarsest level data
            <options>: Dictionary of Training options
                <n_neighbors>: How many nearest neighbors to select
            <level>: Which level of refinement we are at
            <nresult1>: Nearest neighbors of negative data at this refinement level
            <presult1>: Nearest neighbors of positive data at this refinement level
            <nresult>: Nearest neighbors of negative data at previous refinement level
            <presult>: Nearest neighbors of positive data at previous refinement level
            <Model_Selec>: Whether or not to perform hyperparameter tuning
            <Imb_size>: Maximum size of positive and negative data individually
            <coarse>: Binary variable indicating if we have hit the coarsest level.
            <epochs>: Number of epochs to train neural network for
            <Multilevel>: Whether or not we're doing the multilevel version of the code.
            <Depth>: What refinement depth we're at
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
           <Depth>: What refinement depth we're at
           <trainedNetwork>: The trained neural network for this level
           <options>: Dictionary of Hyperparameters
           <Best>: Dictionary of best results found so far
           <flag>: Flag for ceasing refinement if results have not improved in given number of levels
           <Level_results>: Results per refinement level
   '''

    DATA_size = traindata.shape[0]

    # If the combined positive and negative data is below the maximum training threshold, 
    # begin training. 
    if DATA_size < options["Upperlim"] | coarse == 1:

        Depth = level+1

        model = neuralNetwork(traindata, train_l, options)
        Results = Evaluate(model, valdata, val_l)

        # Indicate that training of the coarsest section is done.
        coarse = 0

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
        NfineData = NdataCoarse
        PfineData = PdataCoarse

        # If there are too many points in each class to begin training, begin coarsening
        if NfineData["Data"].shape[0] > options["Imb_size"]:
            NdataCoarse = coarsen(NdataFine, options["n_neighbors"], T=0.6)

        if PfineData["Data"].shape[0] > options["Imb_size"]:
            PdataCoarse = coarsen(PdataFine, options["n_neighbors"], T=0.6)

        traindata = pd.concat([NdataCoarse["Data"], PdataCoarse["Data"]])
        train_l = pd.concat([NdataCoarse["Label"], PdataCoarse["Label"]])

        #Pweight = 1/len(PcoarseLbl,1)
        #Nweight = 1/len(NcoarseLbl,1)

        # If the size of each dataset is considered small enough or no more meaningful coarsening
        # can be performed, then this is the coarsest level of data. 
        if ((NdataCoarse["Data"].shape[0] < options["Imb_size"]) & (PdataCoarse["Data"].shape[0] < options["Imb_size"])) | \
                (NdataCoarse["Data"].shape[0] == NdataFine["Data"].shape[0]):
            coarse = 1

        # Go to next iteration of recursion
        Results, posBorderData, negBorderData, Depth, options, Best, flag, Level_results = MLD(
            traindata, train_l, valdata, val_l, level, NdataCoarse, PdataCoarse, options, Depth,
            NdataFine, PdataFine, coarse)


        # Once all of the coarsening has been performed, begin refining the dataset
        traindata, train_l = refine(trainedNetwork, traindata, train_l, numBorderPoints)

        #
        model = neuralNetwork()

        """
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
    """
        flag=1
        Level_results=None

    return model, posBorderData, negBorderData, Depth, options, Best, flag, Level_results