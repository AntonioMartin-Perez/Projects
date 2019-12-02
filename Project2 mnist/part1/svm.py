import numpy as np
from sklearn.svm import LinearSVC
# from sklearn.metrics import zero_one_loss


### Functions for you to fill in ###

#pragma: coderesponse template
def one_vs_rest_svm(train_x, train_y, test_x):
    """
    Trains a linear SVM for binary classifciation

    Args:
        train_x - (n, d) NumPy array (n datapoints each with d features)
        train_y - (n, ) NumPy array containing the labels (0 or 1) for each training data point
        test_x - (m, d) NumPy array (m datapoints each with d features)
    Returns:
        pred_test_y - (m,) NumPy array containing the labels (0 or 1) for each test data point
    """
    model = LinearSVC(random_state = 0 , C = 0.1)
    model.fit( train_x , train_y )
    return model.predict(test_x)
    raise NotImplementedError
#pragma: coderesponse end

# n, m, d = 5, 3, 7
# train_x = np.random.random((n, d))
# test_x = train_x[:m]
# train_y = np.zeros(n)
# train_y[-1] = 1
# exp_res = np.zeros(m)
# print(one_vs_rest_svm   (train_x, train_y, test_x)  )


#pragma: coderesponse template
def multi_class_svm(train_x, train_y, test_x):
    """
    Trains a linear SVM for multiclass classifciation using a one-vs-rest strategy

    Args:
        train_x - (n, d) NumPy array (n datapoints each with d features)
        train_y - (n, ) NumPy array containing the labels (int) for each training data point
        test_x - (m, d) NumPy array (m datapoints each with d features)
    Returns:
        pred_test_y - (m,) NumPy array containing the labels (int) for each test data point
    """
    model = LinearSVC(random_state = 0, C = 0.1 )
    model.fit( train_x , train_y )
    return model.predict(test_x)
#pragma: coderesponse end


def compute_test_error_svm(test_y, pred_test_y):
    return 1 - np.mean(pred_test_y == test_y)
    # raise NotImplementedError

