import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# evaluate_std: evaluates Accuracy for the categorical vars
# and MSE for the regression vars
# -----------------------------------------------------------

def evaluate_std(modelRegLong, modelRegLat, modelClfBuild, modelClfFloor, xDev, yDev, scalerLong, scalerLat):
    errLong = abs((scalerLong.inverse_transform(modelRegLong.predict(xDev).flatten().reshape(len(xDev),1)).T.flatten()) - yDev['LONGITUDE'])
    errLat = abs((scalerLat.inverse_transform(modelRegLat.predict(xDev).flatten().reshape(len(xDev),1)).T.flatten()) - yDev['LATITUDE'])

    errCord = np.mean( np.sqrt( errLong**2 + errLat**2 ) )
    errLong = np.mean(errLong)
    errLat = np.mean(errLat)

    AccFloor = np.sum( np.argmax(modelClfFloor.predict(xDev), axis=1) == yDev['FLOOR'].values )/len(yDev)
    AccBuild = np.sum( np.argmax(modelClfBuild.predict(xDev), axis=1) == yDev['BUILDINGID'].values )/len(yDev)

    print("Mean absolute error in Longitude: {:.2f} m".format(errLong))
    print("Mean absolute error in Latitude: {:.2f} m".format(errLat))
    print("Mean squared error in Co-ordinates: {:.2f} m".format(errCord))

    print("Accuracy in predicting Building ID: {:.2f} %".format(AccBuild*100))
    print("Accuracy in predicting Floor: {:.2f} %".format(AccFloor*100))

    return


# -----------------------------------------------------------
# plotAccLoss: plots loss and val_loss for trained models
# -----------------------------------------------------------

def plotAccLoss(hist):
    # Retrieve a list of accuracy results on training and validation data
    # sets for each training epoch
    try:
        acc = hist.history['acc']
        val_acc = hist.history['val_acc']
        # Get number of epochs
        epochs = range(len(acc))
        # Plot training and validation accuracy per epoch
        plt.plot(epochs, acc)
        plt.plot(epochs, val_acc)
        plt.title('Training and validation accuracy')
    except:
        pass
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    epochs = range(len(loss))
    plt.figure()
    # Plot training and validation loss per epoch
    plt.plot(epochs, loss)
    plt.plot(epochs, val_loss)
    plt.title('Training and validation loss')
    plt.show()
