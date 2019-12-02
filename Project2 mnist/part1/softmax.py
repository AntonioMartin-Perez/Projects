import sys
sys.path.append("..")
import utils
from utils import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse


def augment_feature_vector(X):
    """
    Adds the x[i][0] = 1 feature for each data point x[i].

    Args:
        X - a NumPy matrix of n data points, each with d - 1 features

    Returns: X_augment, an (n, d) NumPy array with the added feature for each datapoint
    """
    column_of_ones = np.zeros([len(X), 1]) + 1
    return np.hstack((column_of_ones, X))

#pragma: coderesponse template
def compute_probabilities(X, theta, temp_parameter):
    """
    Computes, for each datapoint X[i], the probability that X[i] is labeled as j
    for j = 0, 1, ..., k-1

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        theta - (k, d) NumPy array, where row j represents the parameters of our model for label j
        temp_parameter - the temperature parameter of softmax function (scalar)
    Returns:
        H - (k, n) NumPy array, where each entry H[j][i] is the probability that X[i] is labeled as j
    """
    exponente = np.dot( theta , X.T) / temp_parameter
    # print("exponente ", exponente)
    c = np.amax( exponente , axis = 0)
    h0 = np.exp(    exponente  - c  )
    # print("h0 ",h0)
    norm = np.sum(  h0 , axis = 0 )
    # print("norm" ,norm)
    return h0/ norm
    raise NotImplementedError
#pragma: coderesponse end

# n, d, k = 3, 5, 7
# X = np.arange(0, n * d).reshape(n, d)
# # zeros = np.zeros((k, d))
# theta = np.arange(0, k * d).reshape(k, d)
# temp_parameter = 0.2
# print(  compute_probabilities(X,theta,temp_parameter) )

#pragma: coderesponse template
def compute_cost_function_loops(X, Y, theta, lambda_factor, temp_parameter):
    """
    Computes the total cost over every datapoint.

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns
        c - the cost value (scalar)
    """
    term1 = 0
    term2 = 0
    """"""
    for i in range( len(X)  ):
        # print(i)
        for j in range( len(theta) ) :
            # print("i,j: ",i,j)
            # print( ( Y[i] == j )   ) #Puede dar problemas por la indexación de python que comienza en 0

            denominator = 0
            for l in range( len(theta) ) :
                denominator += np.exp(  np.dot(  theta[l],X[i].T  ) / temp_parameter  )
            
            # print("denominador ", denominator)

            term1 += (
                 ( Y[i] == j ) * np.log(
                np.exp( np.dot( theta[j],X[i].T ) / temp_parameter )/(
                    denominator    )                        
                )   
            )
    
    for i in range( len(theta) ) :
        for j in range( len ( X.T) ) :
            term2 += theta[i,j]**2

    return  ( - term1 / len( X ) ) + lambda_factor * term2 /2
    # return term1, term2, ( - term1 / len( X ) ) + lambda_factor * term2 /2
#pragma: coderesponse end

#pragma: coderesponse template
def compute_cost_function_old(X, Y, theta, lambda_factor, temp_parameter):
    """
    Computes the total cost over every datapoint.

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns
        c - the cost value (scalar)
    """
    print("X: ",X)
    print("Y: ",Y)
    print("theta: ",theta)
    print("lambda_factor: ",lambda_factor)
    print("temp_parameter: ",temp_parameter)

    term1 = 0
    term2 = 0
    """"""
    # breakpoint()
    mat_exponents_ini   =   np.dot( theta,X.T) / temp_parameter
    c               =   np.max(mat_exponents_ini)
    mat_exponents   =   mat_exponents_ini - c
    numerator       =   np.exp( mat_exponents   )
    denominator     =   np.sum( numerator , axis = 0)
    # breakpoint()
    row_ind         =   Y.copy() #tendrá ? filas
    col_ind         =   np.array(range(len(Y)))    
    data            =   np.array(   [   1 for i in range(   len(Y)  )  ]   )    
    y_equal_label   =   sparse.csc_matrix((data, (row_ind, col_ind)),shape=(len(Y),len(theta)))
    # En la matriz sparse hay que especificar la forma (shape) ya que al ser 0 los elementos, los obvia. 
    term1 = np.sum(
        #np.sum(
        # np.multiply( y_equal_label.toarray)() , np.log( numerator / denominator ) ),  # Hay que usar la @ para multiplicar matrices dispersa-densa
        y_equal_label.toarray() @ np.log( numerator / denominator ),  # Hay que usar la @ para multiplicar matrices dispersa-densa
        axis=0
    )[0]

    term2 = np.sum(
        np.sum(
           np.square( theta ),
        axis = 1)
    )

    return ( - term1 / len( X ) ) + lambda_factor * term2 /2
    # return term1, term2 , ( - term1 / len( X ) ) + lambda_factor * term2 /2
#pragma: coderesponse end

def compute_cost_function(X, Y, theta, lambda_factor, temp_parameter):
    """
    Computes the total cost over every datapoint.

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns
        c - the cost value (scalar)
    """
    # print("X: ",X)
    # print("Y: ",Y)
    # print("theta: ",theta)
    # print("lambda_factor: ",lambda_factor)
    # print("temp_parameter: ",temp_parameter)

    term1 = 0
    term2 = 0
    """"""
    # breakpoint()
    probabilities = np.clip ( 
        compute_probabilities(X,theta,temp_parameter) ,
        0.0000000000000001,1
    )
    # breakpoint()
    row_ind         =   Y.copy() #tendrá ? filas
    col_ind         =   np.array(range(len(Y)))    
    data            =   np.array(   [   1 for i in range(   len(Y)  )  ]   )    
    # breakpoint()
    y_equal_label   =   sparse.coo_matrix((data, (row_ind, col_ind)),shape=(len(theta),len(Y)))
    # breakpoint()
    term1 = np.sum(
        np.multiply(   y_equal_label.toarray() , np.log(probabilities)   )
    )
    # breakpoint()
    term2 = np.sum(
        np.sum(
           np.square( theta ),
        axis = 1)
    )
    return ( - term1 / len( X ) ) + lambda_factor * term2 /2
    # return term1, term2 , ( - term1 / len( X ) ) + lambda_factor * term2 /2
#pragma: coderesponse end


# n, d, k = 3, 5, 7
# X = np.arange(0, n * d).reshape(n, d)
# Y = np.arange(0, n)
# zeros = np.zeros((k, d))
# theta = zeros.copy()
# temp_parameter = 0.2
# lambda_factor = 0.5
# print( compute_cost_function_loops (X,Y,theta,lambda_factor,temp_parameter) )
# print( compute_cost_function (X,Y,theta,lambda_factor,temp_parameter) )



#pragma: coderesponse template
def run_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter):
    """
    Runs one step of batch gradient descent

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        alpha - the learning rate (scalar)
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        theta - (k, d) NumPy array that is the final value of parameters theta
    """
    # print('X = ', X)
    # print('Y = ', Y)
    # print('theta = ', theta)
    # print('alpha = ', alpha)
    # print('lambda_factor = ', lambda_factor)
    # print('temp_parameter = ', temp_parameter)

    p = np.array([]).astype(float)
    p = compute_probabilities(  X, theta, temp_parameter    )

    # print("p = ", p)

    row_ind         =   Y.copy() #tendrá ? filas
    col_ind         =   np.array(range(len(Y)))    
    data            =   np.array(   [   1 for i in range(   len(Y)  )  ]   )    
    y_equal_label   =   sparse.coo_matrix((data, (row_ind, col_ind)),shape=(len(theta),len(Y)))

    # print("y_equal_label = ",y_equal_label.toarray() )

    term1 = np.array([]).astype(float)
    term1 = np.dot (
            y_equal_label.toarray().astype(float) - p , X
        )

    # print("term1 = ", term1)

    term2 = np.array([]).astype(float)
    term2 = lambda_factor * theta

    # print("term2 = ", term2)

    derivative =  np.array([]).astype(float)
    derivative =  term1 * (-1) / (temp_parameter *len(X)) + lambda_factor * term2
    
    # print("derivative = ", derivative)

    updated_theta = np.array([]).astype(float)
    updated_theta = theta - alpha * derivative  


    return updated_theta#, "Este es mi output"
    # raise NotImplementedError
#pragma: coderesponse end
# n, d, k = 3, 5, 7
# X = np.arange(0, n * d).reshape(n, d)
# Y = np.arange(0, n)
# zeros = np.zeros((k, d))
# theta = zeros.copy()
# alpha = 2
# temp_parameter = 0.2
# lambda_factor = 0.5

# print(run_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter) )
# print( run_gradient_descent_iteration(X, Y, run_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter), alpha, lambda_factor, temp_parameter) )

#pragma: coderesponse template
def update_y(train_y, test_y):
    """
    Changes the old digit labels for the training and test set for the new (mod 3)
    labels.

    Args:
        train_y - (n, ) NumPy array containing the labels (a number between 0-9)
                 for each datapoint in the training set
        test_y - (n, ) NumPy array containing the labels (a number between 0-9)
                for each datapoint in the test set

    Returns:
        train_y_mod3 - (n, ) NumPy array containing the new labels (a number between 0-2)
                     for each datapoint in the training set
        test_y_mod3 - (n, ) NumPy array containing the new labels (a number between 0-2)
                    for each datapoint in the test set
    """
    # print(train_y)
    # print(test_y)
    return train_y %3, test_y %3
    # raise NotImplementedError
#pragma: coderesponse end

# train_y = np.arange(0, 10)
# test_y = np.arange(9, -1, -1)
# exp_res = (
#         np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0]),
#         np.array([0, 2, 1, 0, 2, 1, 0, 2, 1, 0])
#         )
# print(update_y(train_y,test_y))

#pragma: coderesponse template
def compute_test_error_mod3(X, Y, theta, temp_parameter):
    """
    Returns the error of these new labels when the classifier predicts the digit. (mod 3)

    Args:
        X - (n, d - 1) NumPy array (n datapoints each with d - 1 features)
        Y - (n, ) NumPy array containing the labels (a number from 0-2) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        test_error - the error rate of the classifier (scalar)
    """
    error_count = 0.

    test_ymod3 = Y %3
    assigned_labels = get_classification(X, theta, temp_parameter)
    train_ymod3 = assigned_labels %3
    return 1 - np.mean(train_ymod3 == test_ymod3)
#pragma: coderesponse end

def softmax_regression(X, Y, temp_parameter, alpha, lambda_factor, k, num_iterations):
    """
    Runs batch gradient descent for a specified number of iterations on a dataset
    with theta initialized to the all-zeros array. Here, theta is a k by d NumPy array
    where row j represents the parameters of our model for label j for
    j = 0, 1, ..., k-1

    Args:
        X - (n, d - 1) NumPy array (n data points, each with d-1 features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        temp_parameter - the temperature parameter of softmax function (scalar)
        alpha - the learning rate (scalar)
        lambda_factor - the regularization constant (scalar)
        k - the number of labels (scalar)
        num_iterations - the number of iterations to run gradient descent (scalar)

    Returns:
        theta - (k, d) NumPy array that is the final value of parameters theta
        cost_function_progression - a Python list containing the cost calculated at each step of gradient descent
    """
    X = augment_feature_vector(X)
    theta = np.zeros([k, X.shape[1]])
    cost_function_progression = []
    for i in range(num_iterations):
        cost_function_progression.append(compute_cost_function(X, Y, theta, lambda_factor, temp_parameter))
        theta = run_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter)
    return theta, cost_function_progression

def softmax_regression_no_augment(X, Y, temp_parameter, alpha, lambda_factor, k, num_iterations):
    """
    Runs batch gradient descent for a specified number of iterations on a dataset
    with theta initialized to the all-zeros array. Here, theta is a k by d NumPy array
    where row j represents the parameters of our model for label j for
    j = 0, 1, ..., k-1

    Args:
        X - (n, d - 1) NumPy array (n data points, each with d-1 features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        temp_parameter - the temperature parameter of softmax function (scalar)
        alpha - the learning rate (scalar)
        lambda_factor - the regularization constant (scalar)
        k - the number of labels (scalar)
        num_iterations - the number of iterations to run gradient descent (scalar)

    Returns:
        theta - (k, d) NumPy array that is the final value of parameters theta
        cost_function_progression - a Python list containing the cost calculated at each step of gradient descent
    """
    # X = augment_feature_vector(X)
    theta = np.zeros([k, X.shape[1]])
    cost_function_progression = []
    for i in range(num_iterations):
        cost_function_progression.append(compute_cost_function(X, Y, theta, lambda_factor, temp_parameter))
        theta = run_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter)
    return theta, cost_function_progression



def get_classification(X, theta, temp_parameter):
    """
    Makes predictions by classifying a given dataset

    Args:
        X - (n, d - 1) NumPy array (n data points, each with d - 1 features)
        theta - (k, d) NumPy array where row j represents the parameters of our model for
                label j
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        Y - (n, ) NumPy array, containing the predicted label (a number between 0-9) for
            each data point
    """
    X = augment_feature_vector(X)
    probabilities = compute_probabilities(X, theta, temp_parameter)
    return np.argmax(probabilities, axis = 0)

def plot_cost_function_over_time(cost_function_history):
    plt.plot(range(len(cost_function_history)), cost_function_history)
    plt.ylabel('Cost Function')
    plt.xlabel('Iteration number')
    plt.show()

def compute_test_error(X, Y, theta, temp_parameter):
    error_count = 0.
    assigned_labels = get_classification(X, theta, temp_parameter)
    return 1 - np.mean(assigned_labels == Y)
