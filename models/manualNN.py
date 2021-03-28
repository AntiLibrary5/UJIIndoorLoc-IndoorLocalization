import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# Implements a manually coded single node NN for reg and clf
# -----------------------------------------------------------

def affine_transform(W, b, x):
    # Input:
    # - W: projection matrix
    # - b: bias
    # - x: input features
    # Output:
    # - vector Wx+b
    v = W@x + b
    return v


def backward_affine_transform(W, b, x, g):
    # Input:
    # - W: projection matrix
    # - b: bias
    # - x: input features
    # - g: incoming gradient
    # Output:
    # - g_W: gradient wrt W of Wx+b
    # - g_b: gradient wrt b of Wx+b
    g_W = np.outer(g,x)
    g_b = g
    return g_W, g_b


def softmax(x):
    # Input:
    # - x: vector of logits
    # Output
    # - vector of probabilities
    x = np.array(x)
    b = x.max()
    x = x - b #to deal with nan
    return np.exp(x)/ np.sum(np.exp(x))


def nll(x, gold):
    # Input:
    # - x: vector of logits
    # - gold: index of the gold class
    # Output:
    # - scalar equal to the Negative log-likelihood
    return -np.log(softmax(x)[gold])


def backward_nll(x, gold, g):
    # Input:
    # - x: vector of logits
    # - gold: index of the gold class
    # - gradient (scalar)
    # Output:
    # - gradient of NLL wrt x
    line = np.zeros(x.shape)
    line[gold] = -1
    g_x = line + ( np.exp(x) / np.sum(np.exp(x)) )
    return g_x * g


def mse(x, gold):
    # Input:
    # - x: vector of logits
    # - gold: index of the gold class
    # Output:
    # - scalar: Mean squared Error
    return np.mean( (x-gold)**2 )


def backward_mse(x, gold, g):
    # Input:
    # - x: vector of logits
    # - gold: index of the gold class
    # - gradient (scalar)
    # Output:
    # - gradient wrt x of MSE
    g_x =  2*np.mean(x-gold)
    return g_x * g


def zero_init(b):
    # Input:
    # - b: bias
    # Output:
    # - zeros init
    b[:] = 0.


def glorot_init(W):
    # Input:
    # - W: weights
    # Output:
    # - zeros in
    n = len(W[0])
    lb = -(np.sqrt(6)/np.sqrt(n))
    ub = (np.sqrt(6)/np.sqrt(n))
    return np.random.uniform(lb,ub,size=W.shape)


def create_parameters(dim_input, dim_output):
    # Input:
    # dim_input: X_train.shape[1]
    # dim_output: number of classes for classification; 1 for reg
    W = glorot_init(np.zeros((dim_output, dim_input)))
    b = np.zeros(dim_output)
    return W, b


def print_n_parameters(W, b):
    # Input: W, b weights and bias
    print("Number of parameters: ", (W.shape[0]*W.shape[1] + b.shape[0]))


def init_weights(dim_input, dim_output):
    # Input:
    # dim_input: X_train.shape[1]
    # dim_output: number of classes for classification; 1 for reg
    # Output: W, b weights and bias
    W, b = create_parameters(dim_input, dim_output)
    zero_init(b)
    glorot_init(W)
    print_n_parameters(W, b)
    return W, b


def nn(X_train, Y_train, X_dev, Y_dev, W, b, n_epochs, step, clf):
    # Input:
    # X_train: input train features
    # Y_train: target train features
    # X_dev: test input features
    # Y_dev: test target features
    # W: initial weight params
    # b: initial bias params
    # n_epochs: number of epochs to be run
    # clf: bool. 0 for regression; 1 for Classification
    # Output:
    # W, b: trained parameters
    Loss_train = []
    Loss_dev = []
    Acc_train = []
    Acc_dev = []
    if clf == 1:
        print("Training a classification model...")
    else:
        print("Training a regression model...")
    for epoch in range(n_epochs):
        y_dev = []
        y_train = []
        loss_train = []
        loss_dev = []
        for counter in range(X_train.shape[0]):
            data_train = X_train.iloc[counter].values
            gold = Y_train[counter]
            x = affine_transform(W, b, data_train)
            if clf==1:
                g_y = backward_nll(x, gold, 1.)
                loss_train.append(nll(x, gold))
                y_train.append( np.argmax(softmax(x)) == gold )
            else:
                g_y = backward_mse(x, gold, 1.)
                loss_train.append(mse(x, gold))
                y_train.append( abs(x-gold) )
            g_W, g_b = backward_affine_transform(W, b, data_train, g_y)
            W = W - step * g_W
            b = b - step * g_b
        for i in range(X_dev.shape[0]):
            if clf==1:
                data_dev = X_dev.iloc[i].values
                gold = Y_dev[i]
                x = affine_transform(W, b, data_dev)
                loss_dev.append(nll(x, gold))
                y_dev.append( np.argmax(softmax(x)) == gold )
        if clf==1:
            Acc_dev.append( np.sum(y_dev)/len(y_dev) )
            Acc_train.append( np.sum(y_train)/len(y_train) )
            Loss_train.append( np.mean(loss_train) )
            Loss_dev.append( np.mean(loss_dev) )
        else:
            Acc_dev.append( np.sum(y_dev)/len(y_dev) )
            Acc_train.append( np.mean(y_train) )
            Loss_train.append( np.mean(loss_train) )
            Loss_dev.append( np.mean(loss_dev) )
        print("Epoch: {}".format(epoch+1))
        if epoch > 2:
            if abs(Acc_dev[-1] - Acc_dev[-2]) < 0.0001:
                print("Stopping...")
                break
    if clf==1:
        plt.figure()
        plt.plot(range(epoch+1), Acc_train, label='Train acc')
        plt.plot(range(epoch+1), Acc_dev, label='Dev acc', marker='*')
        plt.legend()
        plt.show()
    else:
        plt.figure()
        plt.plot(range(n_epochs), Acc_train, label='Discrepancy')
        plt.xlabel('epochs')
        plt.ylabel('Discrepancy in UTM [m]')
        plt.show()
        print("Mean discrepancy: ", Acc_train[-1])
    return W, b


def Run_manualNN(n_epochs, step_size, X_train, Y_train, X_test, Y_test):
    n_epochs = n_epochs #number of epochs
    step = step_size #step size for gradient updates
    # Train model to predict Building ID
    W, b = init_weights(X_train.shape[1], 3) #before training, we initialize the parameters of the network
    W_Build, b_Build = nn(X_train, Y_train['BUILDINGID'], X_test, Y_test['BUILDINGID'], W, b, n_epochs, step_size, clf=1)

    # Train model to predict Floor
    W, b = init_weights(X_train.shape[1], 5) #before training, we initialize the parameters of the network
    W_Floor, b_Floor = nn(X_train, Y_train['FLOOR'], X_test, Y_test['FLOOR'], W, b, n_epochs, step_size, clf=1)

    # Train model to predict Longitude
    W, b = init_weights(X_train.shape[1], 1) #before training, we initialize the parameters of the network
    W_Long, b_Long = nn(X_train, Y_train['LONGITUDE'], X_test, Y_test['LONGITUDE'], W, b, n_epochs, step_size, clf=0)

    # Train model to predict Latitude: REGRESSION
    W, b = init_weights(X_train.shape[1], 1) #before training, we initialize the parameters of the network
    W_Lat, b_Lat = nn(X_train, Y_train['LATITUDE'], X_test, Y_test['LATITUDE'], W, b, n_epochs, step_size, clf=0)

    return W_Build, b_Build, W_Floor, b_Floor, W_Long, b_Long, W_Lat, b_Lat
