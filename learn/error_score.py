import numpy as np

def mae_score(y_actual, y_predic):
    if type(y_actual) != type(np.array([])):
        y_actual = np.array(y_actual)
    if type(y_predic) != type(np.array([])):
        y_predic = np.array(y_predic)
        
    errorpredic = y_actual - y_predic
    
    return (1/len(y_actual)) * sum(errorpredic)
    

def mse_score(y_actual, y_predic):
    if type(y_actual) != type(np.array([])):
        y_actual = np.array(y_actual)
    if type(y_predic) != type(np.array([])):
        y_predic = np.array(y_predic)
        
    errorpredic = (y_actual - y_predic)**2
    
    return (1/len(y_actual)) * sum(errorpredic)

def r2_score(y_actual, y_predic):
    if type(y_actual) != type(np.array([])):
        y_actual = np.array(y_actual)
    if type(y_predic) != type(np.array([])):
        y_predic = np.array(y_predic)
    
    yBar = np.average(y_actual)
    # up and down
    numerator = sum( (y_actual - y_predic)**2 )
    denominator = sum( (y_actual - yBar)**2 )
    
    r2_score = 1 - (numerator/denominator)
    
    return r2_score
    
    
    