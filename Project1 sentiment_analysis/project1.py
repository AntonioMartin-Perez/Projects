from string import punctuation, digits
import numpy as np
import random

# Part I


#pragma: coderesponse template
def get_order(n_samples):
    try:
        with open(str(n_samples) + '.txt') as fp:
            line = fp.readline()
            return list(map(int, line.split(',')))
    except FileNotFoundError:
        random.seed(1)
        indices = list(range(n_samples))
        random.shuffle(indices)
        return indices
#pragma: coderesponse end


#pragma: coderesponse template
def hinge_loss_single(feature_vector, label, theta, theta_0):
    """
    Finds the hinge loss on a single data point given specific classification
    parameters.

    Args:
        feature_vector - A numpy array describing the given data point.
        label - A real valued number, the correct classification of the data
            point.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.


    Returns: A real number representing the hinge loss associated with the
    given data point and parameters.
    """
    return np.maximum(0,1-label*(np.dot(theta,feature_vector)+theta_0))
    # raise NotImplementedError
#pragma: coderesponse end


#pragma: coderesponse template
def hinge_loss_full(feature_matrix, labels, theta, theta_0):
    # print('labels = ',labels)
    """
    Finds the total hinge loss on a set of data given specific classification
    parameters.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.


    Returns: A real number representing the hinge loss associated with the
    given dataset and parameters. This number should be the average hinge
    loss across all of the points in the feature matrix.
    """
    # Your code here
    total = 0.
    i = 0 #counter
    for x in feature_matrix:
        # print(i)
        # print("feature vector = ",x)
        # print( "hinge loss = ",hinge_loss_single(x,labels[i],theta,theta_0) )
        # print('labels i-ésimo = ',labels[i])
        total   +=  hinge_loss_single(x,labels[i],theta,theta_0)
        # print("total = ",total)
        i += 1
    Loss_average = total / len(feature_matrix)
    return Loss_average
    # raise NotImplementedError
#pragma: coderesponse end

# feature_matrix = np.array([[1, 2], [1, 3],[1,-1]])
# label, theta, theta_0 = np.array([-2, -1,1]), np.array([1, 1]), -0.2
# print( hinge_loss_full(feature_matrix,label,theta,theta_0) )


# # feature_matrix = np.array([[0.51809794,0.79741881,0.80974009,0.07246738,0.03129449,0.74946744,0.81546969,0.90935509,0.45663913,0.39322104]
# # ,[0.9573838,0.56345003,0.82888278,0.52842904,0.65315491,0.28461431,0.16599566,0.85502651,0.29627178,0.06996738]]) #np.random.random([3,3])
# labels = np.array([0,2])
# theta = np.array([0.99527124,0.08359483,0.12208552,0.04749423,0.51732139,0.92165108
# ,0.76683263,0.12864038,0.04488454,0.84351819])
# theta_0 = 0

# print( hinge_loss_full(feature_matrix,labels,theta,theta_0) )


#pragma: coderesponse template
def perceptron_single_step_update(
        feature_vector,
        label,
        current_theta,
        current_theta_0):
    """
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the perceptron algorithm.

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        current_theta - The current theta being used by the perceptron
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the perceptron
            algorithm before this update.

    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.
    """
    # Your code here
    if label * (  np.dot(current_theta,feature_vector)  + current_theta_0) <= 0:
        current_theta   = current_theta +  np.float(label) * feature_vector
#        current_theta   +=  np.multiply( label , feature_vector ,out = feature_vector, casting = "unsafe")
        current_theta_0 +=  label
    return current_theta, current_theta_0
    # raise NotImplementedError
#pragma: coderesponse end
# feature_vector = np.array([1, 2])
# label, theta, theta_0 = 1, np.array([-1, 1]), -1.5
# print( perceptron_single_step_update(feature_vector, label, theta, theta_0)  )


#pragma: coderesponse template
def perceptron(feature_matrix, labels, T):
    """
    Runs the full perceptron algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    NOTE: Iterate the data matrix by the orders returned by get_order(feature_matrix.shape[0])

    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns: A tuple where the first element is a numpy array with the value of
    theta, the linear classification parameter, after T iterations through the
    feature matrix and the second element is a real number with the value of
    theta_0, the offset classification parameter, after T iterations through
    the feature matrix.
    """
    # Your code here
    theta, theta_0 =  np.array([0]*len(feature_matrix[0])), 0
    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            # print(feature_matrix[i])
            theta, theta_0 = perceptron_single_step_update(feature_matrix[i],labels[i],theta,theta_0)
            # pass
    # raise NotImplementedError
    return theta,theta_0
#pragma: coderesponse end

# feature_matrix = np.array([[1, 2], [-1, 0]])
# labels = np.array([1, 1])
# T = 1


#pragma: coderesponse template
def average_perceptron(feature_matrix, labels, T):
    """
    Runs the average perceptron algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    NOTE: Iterate the data matrix by the orders returned by get_order(feature_matrix.shape[0])


    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns: A tuple where the first element is a numpy array with the value of
    the average theta, the linear classification parameter, found after T
    iterations through the feature matrix and the second element is a real
    number with the value of the average theta_0, the offset classification
    parameter, found after T iterations through the feature matrix.

    Hint: It is difficult to keep a running average; however, it is simple to
    find a sum and divide.
    """
    # Your code here
    theta, theta_0  =   np.array([0.]*len(feature_matrix[0])), 0
    sum_theta       =   theta.copy()
    sum_theta_0     =   0.
    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            # print(feature_matrix[i])
            theta, theta_0  =   perceptron_single_step_update(feature_matrix[i],labels[i],theta,theta_0)
            sum_theta       +=  theta
            # print(sum_theta)
            sum_theta_0     +=  theta_0
            # print(sum_theta_0)
    # raise NotImplementedError
    return sum_theta/( len(feature_matrix)*T ), sum_theta_0/( len(feature_matrix)*T )
#pragma: coderesponse end
# feature_matrix = np.array([[1, 2]])
# labels = np.array([1])
# T = 1
# print( average_perceptron(feature_matrix, labels, T)   )

#pragma: coderesponse template
def pegasos_single_step_update(
        feature_vector,
        label,
        L,
        eta, 
        current_theta,
        current_theta_0):
    """
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the Pegasos algorithm

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        L - The lamba value being used to update the parameters.
        eta - Learning rate to update parameters.
        current_theta - The current theta being used by the Pegasos
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the
            Pegasos algorithm before this update.

    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.
    """
    if label * (  np.dot(current_theta , feature_vector)  + current_theta_0 ) <= 1:
        current_theta   =  (1 - eta * L)*current_theta + eta * label * feature_vector
        current_theta_0 +=  eta * label
    else:
        current_theta = (1 - eta * L) *current_theta
    return current_theta, current_theta_0    
    # raise NotImplementedError
#pragma: coderesponse end

# feature_vector = np.array([1, 2])
# label, theta, theta_0 = 1, np.array([-1, 1]), -1.5
# L = 0.2
# eta = 0.1
# print(pegasos_single_step_update(feature_vector, label, L, eta, theta, theta_0))


#pragma: coderesponse template
def pegasos(feature_matrix, labels, T, L):
    """
    Runs the Pegasos algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    For each update, set learning rate = 1/sqrt(t),
    where t is a counter for the number of updates performed so far (between 1
    and nT inclusive).

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the algorithm
            should iterate through the feature matrix.
        L - The lamba value being used to update the Pegasos
            algorithm parameters.

    Returns: A tuple where the first element is a numpy array with the value of
    the theta, the linear classification parameter, found after T
    iterations through the feature matrix and the second element is a real
    number with the value of the theta_0, the offset classification
    parameter, found after T iterations through the feature matrix.
    """
    # Your code here
    theta, theta_0 =  np.array([0]*len(feature_matrix[0]) , dtype = float), 0
    contador = 1
    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            # print("feature vector =", feature_matrix[i])
            # print ("thetas ",theta, theta_0)
            eta = 1 / ( np.sqrt(contador))
            # print("eta =",eta)
            theta, theta_0 = pegasos_single_step_update(
                feature_matrix[i],
                labels[i], 
                L ,
                eta,
                theta,
                theta_0)
            contador+=1                       
    # raise NotImplementedError
    return theta,theta_0

    # raise NotImplementedError
#pragma: coderesponse end

# feature_matrix = np.array([[1, 1], [1, 1]])
# labels = np.array([1, 1])
# T = 1
# L = 1
# print(pegasos(feature_matrix, labels, T, L))
# print(      pegasos_single_step_update (feature_matrix[0],labels[0],L,1,np.array([0]*len(feature_matrix[0])), 0 )         )
# print(      pegasos_single_step_update (feature_matrix[0],labels[0],L,1,np.array([1, 1]), 1 )         )



# Part II


#pragma: coderesponse template
def classify(feature_matrix, theta, theta_0):
    """
    A classification function that uses theta and theta_0 to classify a set of
    data points.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
                theta - A numpy array describing the linear classifier.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.

    Returns: A numpy array of 1s and -1s where the kth element of the array is
    the predicted classification of the kth row of the feature matrix using the
    given theta and theta_0. If a prediction is GREATER THAN zero, it should
    be considered a positive classification.
    """
    # Your code here
    labels = np.array([])
    for feature_vector in feature_matrix:
        # breakpoint()
        if np.dot( theta,feature_vector) + theta_0 > 0:
            labels = np.append( labels , [1]  )
        else:
            labels = np.append( labels , [-1] )
    return labels
    # raise NotImplementedError
#pragma: coderesponse end
# feature_matrix = np.array([[1, 1], [1, 1], [1, 1]])
# theta = np.array([1, 1])
# theta_0 = 0

# print(classify (feature_matrix, theta, theta_0 ))

#pragma: coderesponse template
def accuracy(preds, targets):
    """
    Given length-N vectors containing predicted and target labels,
    returns the percentage and number of correct predictions.
    """
    return (preds == targets).mean()
#pragma: coderesponse end


def classifier_accuracy(
        classifier,
        train_feature_matrix,
        val_feature_matrix,
        train_labels,
        val_labels,
        **kwargs):
    """
    Trains a linear classifier and computes accuracy.
    The classifier is trained on the train data. The classifier's
    accuracy on the train and validation data is then returned.

    Args:
        classifier - A classifier function that takes arguments
            (feature matrix, labels, **kwargs) and returns (theta, theta_0)
        train_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        val_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        train_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the training
            feature matrix.
        val_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the validation
            feature matrix.
        **kwargs - Additional named arguments to pass to the classifier
            (e.g. T or L)

    Returns: A tuple in which the first element is the (scalar) accuracy of the
    trained classifier on the training data and the second element is the
    accuracy of the trained classifier on the validation data.
    """
    # Train the classifier on the training set
    if len(kwargs) <= 1:
        train_theta, train_theta_0 = classifier(train_feature_matrix, train_labels, kwargs["T"])
    
    else: #classifier == pegasos:
        train_theta, train_theta_0 = classifier( train_feature_matrix, train_labels, kwargs["T"], kwargs["L"])
    ###########################################################################
    
    #label the training set with the resulting theta, theta_0
    computed_train_clasification = classify( train_feature_matrix, train_theta, train_theta_0)
    # print(computed_train_clasification)

    #Compute trainning accuracy
    train_acc = accuracy ( computed_train_clasification , train_labels )
    # print(train_acc)

    #label the validation set with the resulting theta, theta_0
    computed_val_clasification = classify ( val_feature_matrix, train_theta, train_theta_0 )
    # print(computed_val_clasification)

    #Compute validation accuracy
    # breakpoint()
    val_acc = accuracy (computed_val_clasification, val_labels)
    # print(val_acc)

    # print (train_labels)
    # print("theta = ",train_theta,"theta_0 = ", train_theta_0,)
    return  train_acc, val_acc
    # raise NotImplementedError
#pragma: coderesponse end

# train_feature_matrix = np.array([[1, 0], [1, -1], [2, 3]])
# val_feature_matrix = np.array([[1, 1], [2, -1]])
# train_labels = np.array([1, -1, 1])
# val_labels = np.array([-1, 1])
# T = 1
# L = 0.2
# print(classifier_accuracy (pegasos, train_feature_matrix, val_feature_matrix, train_labels,val_labels, T=3 , L = 0.2 ))
# print(pegasos (feature_matrix, labels,  T=3 , L = 0.2 ))
#pragma: coderesponse template


def extract_words(input_string):
    """
    Helper function for bag_of_words()
    Inputs a text string
    Returns a list of lowercase words in the string.
    Punctuation and digits are separated out into their own words.
    """
    for c in punctuation + digits:
        input_string = input_string.replace(c, ' ' + c + ' ')

    return input_string.lower().split()
#pragma: coderesponse end


#pragma: coderesponse template
def bag_of_words(texts):
    """
    Inputs a list of string reviews
    Returns a dictionary of unique unigrams occurring over the input

    Feel free to change this code as guided by Problem 9
    """
    f = open("stopwords.txt", "r")
    stopwords_with_newlines = f.read()
    stopwords = stopwords_with_newlines.split ("\n")
    # Your code here
    dictionary = {} # maps word to unique index
    for text in texts:
        word_list = extract_words(text)
        for word in word_list:
            if word not in dictionary and word not in stopwords:
                dictionary[word] = len(dictionary)
    return dictionary
#pragma: coderesponse end


#pragma: coderesponse template
def extract_bow_feature_vectors(reviews, dictionary):
    """
    Inputs a list of string reviews
    Inputs the dictionary of words as given by bag_of_words
    Returns the bag-of-words feature matrix representation of the data.
    The returned matrix is of shape (n, m), where n is the number of reviews
    and m the total number of entries in the dictionary.

    Feel free to change this code as guided by Problem 9
    """
    # Your code here

    num_reviews = len(reviews)
    feature_matrix = np.zeros([num_reviews, len(dictionary)])

    for i, text in enumerate(reviews):
        word_list = extract_words(text)
        for word in word_list:
            if word in dictionary:
                feature_matrix[i, dictionary[word]] += 1
    return feature_matrix
#pragma: coderesponse end


#pragma: coderesponse template
################################################# Subí arriba la definición para que no diese problemas al ejecutar en terminal
# def accuracy(preds, targets):
#     """
#     Given length-N vectors containing predicted and target labels,
#     returns the percentage and number of correct predictions.
#     """
#     return (preds == targets).mean()
# #pragma: coderesponse end
