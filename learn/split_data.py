from copy import deepcopy
import numpy as np
from math import floor, ceil

def __randomrange(start, end):
    return np.random.randint(start, end)

def __split_process(X, y, length_split, shuffle, ix, ix_done):
    X_result, y_result = [], []
    
    data_length = len(X)
    
    for test in range(length_split):
        if shuffle is True:
            ix = __randomrange(0, data_length)
            while(ix in ix_done):
                ix = __randomrange(0, data_length)
        else:
            ix += 1
        
        # X_result.append(X[ix])
        X_temp = X[ix].tolist()
        if type(X_temp) is list:
            X_temp = X_temp[0]
        X_result.append(X_temp)
        y_result.append(y[ix])
        
        ix_done.append(ix)
    
    return X_result, y_result

def split(X, y, test_size = 0.2, shuffle = True, sort = False):
    X_train, X_test, y_train, y_test = [], [], [], []
    
    if sort is True:
        X = np.sort(X)
        y = np.sort(y)
    
    data_length = len(X)
    data_test_length = floor(data_length * test_size)
    data_train_length = ceil(data_length - data_test_length)
    
    data_X = deepcopy(X)
    data_y = deepcopy(y)
    
    # ix_done ==> save the index that has been used
    ix_done = []
    ix = -1
    
    X_train, y_train = __split_process (
        data_X, data_y,
        data_train_length, 
        shuffle,
        ix, ix_done)

    ix = len(ix_done)-1
    X_test, y_test = __split_process (
        data_X, data_y,
        data_test_length, 
        shuffle,
        ix, ix_done)
    
    return X_train, X_test, y_train, y_test

