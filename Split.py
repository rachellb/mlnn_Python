from sklearn.model_selection import train_test_split

def Split(data, prop):
    '''
    Inputs:
        <data> : (pd.DataFrame). A dataframe consisting of rows of sample data and their corresponding labels. 
                 Last column must be named "Label"
        <prop> : (float). The proportion of data that is to be considered for training. 
    '''
    # Split labels from data
    Pdata = data[data["Label"] == 1]
    Ndata = data[data["Label"] == 0]

    Ptraindata, Ptestdata = train_test_split(Pdata, train_size=prop)
    Ptrain_lbl = Ptraindata["Label"]
    Ptest_lbl = Ptestdata["Label"]
    Ptraindata = Ptraindata.drop(["Label"], axis=1)
    Ptestdata = Ptestdata.drop(["Label"], axis=1)


    
    return train_lbl, traindata, test_lbl, testdata

