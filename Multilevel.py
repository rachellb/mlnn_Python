import pandas as pd
from MLD import MLD
from NearestNeighborSearch import NearestNeighborSearch
from sklearn.model_selection import train_test_split
#import neuralNetwork
import time
import numpy as np
from neuralNetwork import neuralNetwork
from Evaluate import Evaluate
from testNeuralNetwork import testNetwork
from averageResults import averageResults
import tensorflow as tf
import keras
import pathlib # Create director if it does not exist (requires python 3.5+)


def Multilevel(data, dataName, max_ite=1, prop=0.8, multilevel=1, n_neighbors=10,
               Upperlim=500, Imb_size=300, Model_Selec=1, numBorderPoints=10,  loss="cross", alpha=0.5, gamma=4,
               Level_size=1, refineMethod="border", epochs=100, batch_size=32, Dropout=0.3, batchnorm=0.2, factor=3, patienceLevel=2, weights=False, label=""):

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
    data["Label"] = np.where(data["Label"] == 2, 0, 1)

    # TODO: Remove when finished testing
    # Bad practice, but mostly just to test something
    from sklearn.preprocessing import StandardScaler
    scale = StandardScaler()
    data.iloc[:, 0:-1] = scale.fit_transform(data.iloc[:, 0:-1])

    Results = []
    totalTime = []

    # Model Training Options
    options = {"n_neighbors": n_neighbors, "Upperlim": Upperlim, "Model_Selec": Model_Selec, "Imb_size": Imb_size,
               "loss": loss, "alpha": alpha, "gamma": gamma, "numBorderPoints": numBorderPoints,
               "refineMethod": refineMethod, "epochs": epochs, "batch_size": batch_size, "Dropout": Dropout,
               "BatchNorm": batchnorm,  "factor": factor, "patienceLevel": patienceLevel, "weights": weights,
               "dataName": dataName, "max_ite": max_ite, "multilevel": multilevel}

    for ite in range(1, max_ite+1):
        start = time.time()

        # To make sure not re-running the same neural network
        tf.keras.backend.clear_session()

        # Create train/test data
        traindata, testdata = train_test_split(data, stratify=data["Label"], train_size=0.9)
        # Create train/validation data using the train dataset
        traindata, valdata = train_test_split(traindata, stratify=traindata["Label"], train_size=0.9)

        # Validation is kept separate for determining when to stop refinement
        val_lbl = valdata["Label"]
        valdata = valdata.drop(["Label"], axis=1)

        # Testing data is only used at the end of the algorithm to estimate final performance
        test_lbl = testdata["Label"]
        testdata = testdata.drop(["Label"], axis=1)

        if multilevel == 1:

            Best = {}  # Will contain best results found.

            Ptraindata = traindata[traindata["Label"] == 1]
            Ntraindata = traindata[traindata["Label"] == 0]

            Ptraindata.reset_index(drop=True, inplace=True)
            Ntraindata.reset_index(drop=True, inplace=True)

            Ptrainlbl = Ptraindata["Label"]
            Ntrainlbl = Ntraindata["Label"]

            Ptraindata = Ptraindata.drop(["Label"], axis=1)
            Ntraindata = Ntraindata.drop(["Label"], axis=1)

            # Training data used separately
            train_lbl = traindata["Label"]
            traindata = traindata.drop(["Label"], axis=1)

            nNeighbors = NearestNeighborSearch(Ntraindata, n_neighbors)
            pNeighbors = NearestNeighborSearch(Ptraindata, n_neighbors)

            # Put everything into a dictionary to keep all relevant info together
            negativeData = {"Data": Ntraindata, "Labels": Ntrainlbl, "KNeighbors": nNeighbors}
            positiveData = {"Data": Ptraindata, "Labels": Ptrainlbl, "KNeighbors": pNeighbors}

            level = 0
            model, posBorderData, negBorderData, max_Depth, options, Best, flag, Level_results =\
                MLD(traindata, train_lbl, valdata, val_lbl, level, negativeData, positiveData, options)

            formatFilename = "models/%s/best/"
            filename = formatFilename % (options["dataName"])
            bestModel = keras.models.load_model(filename)

            res = Evaluate(bestModel, testdata, test_lbl)
            Results.append(res)

        else:

            train_lbl = traindata["Label"]
            traindata = traindata.drop(["Label"], axis=1)

            #model = neuralNetwork(traindata, train_lbl, valdata, val_lbl, options)
            model = testNetwork(traindata, train_lbl, valdata, val_lbl, options)

            res = Evaluate(model, testdata, test_lbl)
            Results.append(res)
            max_Depth = 0

        end = time.time()
        totalTime.append(end-start)

    Results = averageResults(Results)
    aveCoarsenDepth = np.mean(max_Depth)
    averageTime = np.mean(totalTime)     
    # Save results into an excel file

    """
    resultsTable = pd.DataFrame({
        'GMean': Results["GMean"],
        'Acc': Results["Acc"],
        'Recall': Results["Recall"],
        'Spec': Results["Spec"],
        #'stdGMean': Results["stdGMean"],
        #'stdAcc': Results["stdAcc"],
        #'stdSen': Results["stdSen"],
        #'stdSpec': Results["stdSpec"],
        'Time (sec)': averageTime, 
        'Average Coarsening Depth': aveCoarsenDepth, 
        #'Refined': Results["Refined"]
    }, columns=['GMean','Acc', 'Recall', 'Spec','stdGMean','stdAcc','stdSen', 'stdSpec', 'Time (sec)', 'Depth', 'Refined'])
    """

    resultsTable = pd.DataFrame({
        'GMean': [Results["GMean"]],
        'Acc': [Results["Acc"]],
        'Recall': [Results["Recall"]],
        'Spec': [Results["Spec"]],
        'Time (sec)': [averageTime],
        'Average Coarsening Depth': [aveCoarsenDepth],
    }, columns=['GMean', 'Acc', 'Recall', 'Spec', 'Time (sec)', 'Depth'])

    formatDirect = "Results/%s/" % options["dataName"]
    pathlib.Path(formatDirect).mkdir(parents=True, exist_ok=True)

    formatFilename = "Multilevel_%depochs%dRefine%sBorderPoints%dNeighbors%dLoss%s%smaxIte%d.xlsx"
    filename = formatFilename % (multilevel, epochs, refineMethod, numBorderPoints, n_neighbors, loss, label, max_ite)
    filename = formatDirect + filename

    resultsTable.to_excel(filename, sheet_name='Sheet1', index=False)


if __name__ == "__main__":
    Multilevel(data="../Hypothyroid.csv", dataName="Hypothyroid", multilevel=1, epochs=1,
               patienceLevel=1, label="amg", max_ite=1, batch_size=256, numBorderPoints=50)
