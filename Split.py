from sklearn.model_selection import train_test_split

def Split(data, prop):
    '''
    Inputs:
        <data> : (pd.DataFrame). A dataframe consisting of rows of sample data and their corresponding labels. 
                 Last column must be named "Label"
        <prop> : (float). The proportion of data that is to be considered for training. 
    '''
    # Split labels from data
    train_data, test_data = train_test_split(data, train_size=prop)
    train_lbl = train_data.pop("Label")
    test_lbl = test_data.pop("Label")
    
    return train_lbl, train_data, test_lbl, test_data

