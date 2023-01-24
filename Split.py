from sklearn.model_selection import train_test_split

def Split(data, prop):
    '''
    data : pd.DataFrame
    prop : float
    '''
    # Split labels from data
    train_data, test_data = train_test_split(data, train_size=prop)
    train_lbl = train_data.pop("Label")
    test_lbl = test_data.pop("Label")
    return train_lbl, train_data, test_lbl, test_data

