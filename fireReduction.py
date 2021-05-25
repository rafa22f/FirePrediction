import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import warnings

def train_one_vs_all(X, y, num_classes, lambda_val):
    '''
    Train a one vs. all logistic regression

    Inputs:
      X                data matrix (2d array shape m x n)
      y                label vector with entries from 0 to
                       num_classes - 1 (1d array length m)
      num_classes      number of classes (integer)
      lambda_val       regularization parameter (scalar)

    Outputs:
      weight_vectors   matrix of weight vectors for each class
                       weight vector for class c in the cth column
                       (2d array shape n x num_classes)
      intercepts       vector of intercepts for all classes
                       (1d array length num_classes)

    '''

    # Write code here

    # Hint: you may find the vector comparison y == i helpful!
    m, n = X.shape
    weight_vectors = np.zeros((n, num_classes))
    intercepts = np.zeros(num_classes)

    for i in range(num_classes):
        yi = (y == i)
        weight_vectors[:,i], intercepts[i] = train_logistic_regression(X, yi, lambda_val)

    return weight_vectors, intercepts


def predict_one_vs_all(X, weight_vectors, intercepts):
    '''
    Train a one vs. all logistic regression

    Inputs:
      X                data matrix (2d array shape m x n)
      weight_vectors   matrix of weight vectors for each class
                       weight vector for class c in the cth column
                       (2d array shape n x num_classes)
      intercepts       vector of intercepts for all classes
                       (1d array length num_classes)

    Outputs:
      predictions      vector of predictions for examples in X
                       (1d array length m)
    '''

    # Write code here

    # Hint: use a matrix vector multiplication to simultaneously make
    # predictions for all classes. Don't forget to add the intercept values

    # Hint: look up the np.argmax function. It can find the index of
    # the largest value in an array, or in each row/column of an array
    vals = X.dot(weight_vectors) + intercepts;
    predictions = np.argmax(vals, axis=1)

    return predictions


def train_logistic_regression(X, y, lambda_val):
    '''
    Train a regularized logistic regression model

    Inputs:
      X           data matrix (2d array shape m x n)
      y           label vector with 0/1 entries (1d array length m)
      lambda_val  regularization parameter (scalar)

    Outputs:
      weights     weight vector (1d array length n)
      intercept   intercept parameter (scalar)
    '''
    model = linear_model.LogisticRegression(C=2./lambda_val, solver='lbfgs')

    # call model.fit(X, y) while suppressing warnings about convergence
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y)

    weight_vector = model.coef_.ravel()
    intercept = model.intercept_
    return weight_vector, intercept

def precisionStart(y, predY):
    '''
    Find the number of true positives and false positives
    for START (class 1) then calculate the precision
    '''
    TP = 0
    FP = 0
    for i in range(y.shape[0]):
        if predY[i] == 1 and y[i] == predY[i]:
            TP += 1
        if predY[i] == 1 and y[i] != predY[i]:
            FP += 1
    if FP == 0:
        FP += 1
    precision = TP/(TP+ FP)


    return precision

def precisionFire(y, predY):
    '''
    Find the number of true positives and false positives
    for START (class 1) then calculate the precision
    '''
    TP = 0
    FP = 0
    for i in range(y.shape[0]):
        if predY[i] == 2 and y[i] == predY[i]:
            TP += 1
        if predY[i] == 2 and y[i] != predY[i]:
            FP += 1
    if FP == 0:
        FP += 1
    precision = TP/(TP+ FP)


    return precision

def precisionNoFire(y, predY):
    '''
    Find the number of true positives and false positives
    for START (class 1) then calculate the precision
    '''
    TP = 0
    FP = 0
    for i in range(y.shape[0]):
        if predY[i] == 0 and y[i] == predY[i]:
            TP += 1
        if predY[i] == 0 and y[i] != predY[i]:
            FP += 1
    if FP == 0:
        FP += 1
    precision = TP/(TP+ FP)


    return precision

#
# y = np.array([1,0,1, 2, 2])
# predY = np.array([1,1,1, 1, 2])
# print(precision(y, predY))
