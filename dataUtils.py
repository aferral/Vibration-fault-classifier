from collections import Counter
import random
import numpy as np
import pylab as plt
from sklearn.decomposition import PCA
import pandas
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

from dataset import Dataset


def flatInput(train_data,train_labels,test_data,test_labels):
    trainX = train_data
    trainY = train_labels

    tx = test_data
    ty = test_labels

    # Flatten data for test images as a vector and labels as numbers
    flattenDataset = np.nan_to_num(np.array([image.flatten() for image in trainX]))
    flatTest = np.nan_to_num(np.array([image.flatten() for image in tx]))
    fTrainLabels = np.where(trainY == 1)[1]
    fTestLabels = np.where(ty == 1)[1]
    return flattenDataset, flatTest, fTrainLabels, fTestLabels

def inverseNorm(image,dataset):
    imsize = dataset.imageSize

    im = image.reshape((imsize,imsize))
    return (im * dataset.std) + dataset.mean

def pca2Visua(pca,data,labels,nClases):
    # Show a few statistics of the data
    print  "Pca with 2 components explained variance " + str(pca.explained_variance_ratio_)
    print "PCA 2 comp of the data (using train)"

    transformed = pca.transform(data)

    plt.figure()
    allscatter = []
    for c in range(nClases):
        elements = np.where(labels == c)
        temp = plt.scatter(transformed[elements, 0], transformed[elements, 1],
                           facecolors='none', label='Class ' + str(c), c=np.random.rand(3, 1))
        allscatter.append(temp)
    plt.legend(tuple(allscatter),
               tuple(["class " + str(c) for c in range(nClases)]),
               scatterpoints=1,
               loc='lower left',
               ncol=3,
               fontsize=8)
    plt.show()


def showRandomImages(dataset,toShow=5):
    # REMEMBER THE IMAGES IN THE DATASET ARE NORMALIZED

    # if you want to see iamges as the files use
    # inverseNorm(img,dataset)
    nClases = dataset.getNclasses()

    dataX = dataset.train_data
    dataY = dataset.train_labels

    for c in range(nClases):
        elements = np.where(dataY == c)


        name = str(c)

        # Show 5 random element of that class
        fig, grid = plt.subplots(1, toShow)
        print "Class ", c
        for j in range(toShow):
            r = random.randint(0, elements[0].shape[0])
            r = r if r <= elements[0].shape[0] else (elements[0].shape[0]-1)
            ind = elements[0][r]

            fileName = dataset.getTrainFilename(ind)
            image = dataX[ind, :,:]

            # If you want to see images as shown in files
            # image = inverseNorm(flattenDataset[ind,:],dataset)

            grid[j].imshow(image)
            grid[j].set_adjustable('box-forced')
            grid[j].autoscale(False)
            grid[j].set_title(str(fileName), fontsize=10)
            grid[j].axis('off')

def showMeanstd(dataset):
    nClases = dataset.getNclasses()
    # Show mean image of train per class
    # NOTE THIS IMAGE IS NOT THE MEAN IMAGE IN DATASET. This mean image is calculated with normalized images
    # THe original mean image was calculated with the original images in train set.
    imsize = dataset.imageSize


    flattenDataset, flatTest, fTrainLabels, fTestLabels = \
        flatInput(dataset.train_data, dataset.train_labels, dataset.test_data, dataset.test_labels)


    fig, grid = plt.subplots(1, nClases)
    for c in range(nClases):
        elements = np.where(fTrainLabels == c)
        classMean = np.mean(flattenDataset[elements, :], axis=1).reshape((imsize, imsize))
        grid[c].set_title('mean class ' + str(c))
        grid[c].imshow(classMean)
        grid[c].set_adjustable('box-forced')
        grid[c].autoscale(False)
        grid[c].axis('off')

    # SHOW IMAGES MEAN AS ORIGINAL IMAGE (INVERT NORM)
    fig, grid = plt.subplots(1, nClases)
    for c in range(nClases):
        elements = np.where(fTrainLabels == c)
        classMean = inverseNorm(np.mean(flattenDataset[elements, :], axis=1), dataset)
        grid[c].set_title('mean Org ' + str(c))
        grid[c].imshow(classMean)
        grid[c].set_adjustable('box-forced')
        grid[c].autoscale(False)
        grid[c].axis('off')

    fig, grid = plt.subplots(1, nClases)
    for c in range(nClases):
        elements = np.where(fTrainLabels == c)
        classMean = np.std(flattenDataset[elements, :], axis=1).reshape((imsize, imsize))
        grid[c].set_title('Std class ' + str(c))
        grid[c].imshow(classMean)
        grid[c].set_adjustable('box-forced')
        grid[c].autoscale(False)
        grid[c].axis('off')

    # SHOW IMAGES MEAN AS ORIGINAL IMAGE (INVERT NORM)
    fig, grid = plt.subplots(1, nClases)
    for c in range(nClases):
        elements = np.where(fTrainLabels == c)
        classMean = inverseNorm(np.std(flattenDataset[elements, :], axis=1), dataset)
        grid[c].set_title('Std Org ' + str(c))
        grid[c].imshow(classMean)
        grid[c].set_adjustable('box-forced')
        grid[c].autoscale(False)
        grid[c].axis('off')

    normMean = np.mean(flattenDataset, axis=0)
    plt.figure()
    plt.title('Mean full train set')
    plt.imshow(dataset.mean)


    plt.figure()
    plt.title('Mean norm full train set  (after invNorm)')
    plt.imshow(inverseNorm(normMean, dataset))


#MODEL TRAIN

def processPca(flattenDataset, flatTest, fTrainLabels, fTestLabels):
    nComp = 6
    pca = PCA(n_components=nComp)
    pca.fit(flattenDataset)
    dataPcaTrainX = pca.transform(flattenDataset)
    dataPcaTrainy = fTrainLabels

    dataPcaTestX = pca.transform(flatTest)
    dataPcaTesty = fTestLabels
    return dataPcaTrainX, dataPcaTestX, dataPcaTrainy, dataPcaTesty


def showDataplots(dx,ltrain):
    data = pandas.DataFrame(data=dx, columns=['pca' + str(i) for i in range(dx.shape[1])])
    data['class'] = pandas.Series(ltrain, index=data.index)
    data.boxplot(by='class')
    sns.pairplot(data, hue="class")


# Use MLP
def runMLP(X, y, tX, ty, returnVal=False):
    clf = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(50, 3), random_state=1)

    clf.fit(X, y)
    pred = clf.predict(tX)

    if not returnVal:
        print classification_report(ty, pred)
    else:
        return accuracy_score(ty, pred)


# Use Linear support machine
def useLinear(X, y, tX, ty, returnVal=False):
    clf = SGDClassifier()
    clf.fit(X, y)
    pred = clf.predict(tX)

    if not returnVal:
        print classification_report(ty, pred)
    else:
        return accuracy_score(ty, pred)


# USE SVM (uses rbf o gaussian kernel)
def useSVM(X, y, tX, ty, returnVal=False):
    clf = SVC()
    clf.fit(X, y)
    pred = clf.predict(tX)

    if not returnVal:
        print classification_report(ty, pred)
    else:
        return accuracy_score(ty, pred)

def getNewDataset(datasetFolder,seed=None):
    if seed is None:
        seed = int(100 * random.random())
    #Changing the seed will give a new train-val-test split
    dataset = Dataset(datasetFolder, batch_size=20,seed=seed)
    flattenDataset, flatTest, fTrainLabels, fTestLabels = flatInput(dataset.train_data, dataset.train_labels, dataset.test_data, dataset.test_labels)
    return flattenDataset, flatTest, fTrainLabels, fTestLabels, dataset




def simplemethodsresults(datasetFolder,transform,limitPoints=None):
    xTimes = 3
    resultMLP = []
    resultLinear = []
    resultSVM = []


    # FOR the 3 alg repeat 10 times
    for i in range(xTimes):
        flattenDataset, flatTest, fTrainLabels, fTestLabels, _ = getNewDataset(datasetFolder)
        if limitPoints is None:
            X, tX, y, ty = transform(flattenDataset, flatTest, fTrainLabels, fTestLabels)
        else:
            X, tX, y, ty = transform(flattenDataset, flatTest, fTrainLabels, fTestLabels)
            X = X[:limitPoints,:]
            y = y[:limitPoints]

        resultMLP.append(runMLP(X, y, tX, ty, True))
        resultLinear.append(useLinear(X, y, tX, ty, True))
        resultSVM.append(useSVM(X, y, tX, ty, True))

    print "Results: "
    print "MLP accuracy mean ", np.mean(np.array(resultMLP)), " std ", np.std(np.array(resultMLP))
    print "Linear SVM accuracy mean ", np.mean(np.array(resultLinear)), " std ", np.std(np.array(resultLinear))
    print "SVM accuracy mean ", np.mean(np.array(resultSVM)), " std ", np.std(np.array(resultSVM))


def count_number_trainable_params(tf):
    '''
    Counts the number of trainable variables.
    '''
    tot_nb_params = 0
    for trainable_variable in tf.trainable_variables():
        shape = trainable_variable.get_shape()  # e.g [D,F] or [W,H,C]
        current_nb_params = get_nb_params_shape(shape)
        print shape, " ", current_nb_params
        tot_nb_params = tot_nb_params + current_nb_params
    print "Total ", tot_nb_params
    return tot_nb_params


def get_nb_params_shape(shape):
    '''
    Computes the total number of params for a given shap.
    Works for any number of shapes etc [D,F] or [W,H,C] computes D*F and W*H*C.
    '''
    nb_params = 1
    for dim in shape:
        nb_params = nb_params * int(dim)
    return nb_params


