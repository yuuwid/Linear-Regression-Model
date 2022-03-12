from copy import deepcopy
import numpy as np


class LinearRegression:
        
    def __init__(self, X_train = [], y_train = [], copy = True):
        self.__copy = copy
        if (copy is True):
            self.__X_train = deepcopy(X_train)
            self.__y_train = deepcopy(y_train)
    
    
    def fit_data(self, X_train, y_train):
        if (self.__copy is True):
            self.__X_train = deepcopy(X_train)
            self.__y_train = deepcopy(y_train)

        self.__calc()
        self.__train()
        
    def __calc(self):
        X = self.__X_train
        y = self.__y_train
        XBar = np.average(X)
        yBar = np.average(y)
        data_length = len(X)
        
        # up and down
        numerator, denominator = 0, 0
        
        for i in range(data_length):
            numerator += (X[i] - XBar) * (y[i] - yBar)
            denominator += ((X[i] - XBar)**2)
            
        a = numerator / denominator
        b = yBar - a * XBar
        
        self.intercept = b
        self.slope = a
        
    def __train(self):
        X = self.__X_train
        y = self.__y_train
        a, b = self.slope,  self.intercept
        
        X_result, y_result = [], []
        self.errortrain = []

        data_length = len(X)
        
        for i in range(data_length):
            y_prediction = self.__linear(a, b, X[i])
            y_actual = y[i]
            
            self.errortrain.append( (y_actual - y_prediction)**2 )
            
            y_result.append(y_prediction)
            X_result.append(X[i])
        
        self.__X_result = X_result
        self.__y_result = y_result

    def __linear(self, a, b, x):
        y = a*x + b
        return y
    
    def trained(self):
        return self.__X_result, self.__y_result
    
    def predic(self, X):
        a, b = self.slope,  self.intercept
        if type(X) != type(np.array([])):
            X = np.array(X)
        result = self.__linear(a, b, X)
        return result.tolist()