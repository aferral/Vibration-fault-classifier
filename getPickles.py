import pickle

import itertools
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',axis=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if axis is None:
        plt.imshow(cm, interpolation='nearest')
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        # plt.show()
    else:
        axis.imshow(cm, interpolation='nearest')
        axis.set_title(title)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            axis.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        axis.set_ylabel('True label')
        axis.set_xlabel('Predicted label')

folder = 'pickledReports'
outName = 'tabla'
listFiles = os.listdir(folder)

df = pd.DataFrame(columns=['Dataset', 'Method', 'accMean', 'accStd', 'runtime'])

names = []
listaConfM =[]
listaConfstd = []
ind = 0
for f in listFiles:

    #Crea tabla
    with open(os.path.join(folder,f),'rb') as of:
        [name, accMean, accStd, meancnfm, stdcnfm, meanTrainTime, resultsRaw, resultsPCA] = pickle.load(of)

        listaConfM.append(meancnfm)
        listaConfstd.append(stdcnfm)

        trozos = name.split(" ")

        dataset = trozos[0]

        names.append(dataset)
        metodo = trozos[1]

        flatD = dataset+"FLAT"
        pcaD = dataset+"PCA"


        df.loc[ind] = pd.Series({'Dataset': dataset, 'Method': metodo,
                                 'accMean': accMean, 'accStd': accStd,'runtime':meanTrainTime})
        ind += 1

        #Add results MLP. LSVM, SVM with flat dataset

        accmMLP = resultsRaw[0][0]
        accstdMLP = resultsRaw[0][1]

        accmLSVM = resultsRaw[1][0]
        accstdLSVM = resultsRaw[1][1]

        accmSVM = resultsRaw[2][0]
        accstdSVM = resultsRaw[2][1]

        df.loc[ind] = pd.Series({'Dataset': flatD, 'Method': "MLP",
                                 'accMean': accmMLP, 'accStd': accstdMLP,'runtime': None})
        ind += 1

        df.loc[ind] = pd.Series({'Dataset': flatD, 'Method': "LSVM",
                                 'accMean': accmLSVM, 'accStd': accstdLSVM,'runtime': None})
        ind += 1

        df.loc[ind] = pd.Series({'Dataset': flatD, 'Method': "SVM",
                                 'accMean': accmSVM, 'accStd': accstdSVM,'runtime': None})
        ind += 1

        # Add results MLP. LSVM, SVM with PCA dataset
        accmMLP = resultsPCA[0][0]
        accstdMLP = resultsPCA[0][1]

        accmLSVM = resultsPCA[1][0]
        accstdLSVM = resultsPCA[1][1]

        accmSVM = resultsPCA[2][0]
        accstdSVM = resultsPCA[2][1]

        df.loc[ind] = pd.Series({'Dataset': pcaD, 'Method': "MLP",
                                 'accMean': accmMLP, 'accStd': accstdMLP, 'runtime': None})
        ind += 1

        df.loc[ind] = pd.Series({'Dataset': pcaD, 'Method': "LSVM",
                                 'accMean': accmLSVM, 'accStd': accstdLSVM, 'runtime': None})
        ind += 1

        df.loc[ind] = pd.Series({'Dataset': pcaD, 'Method': "SVM",
                                 'accMean': accmSVM, 'accStd': accstdSVM, 'runtime': None})
        ind += 1







clases=['c'+str(i) for i in range(listaConfM[0].shape[0])]

fig,axis = plt.subplots(len(listFiles), 2)
for c in range(len(listFiles)):
    mean = listaConfM[c]
    std = listaConfstd[c]

    if len(listFiles) > 1:
        ax1 = axis[c][0]
        ax2 = axis[c][1]
    else:
        ax1 = axis[0]
        ax2 = axis[1]
    plot_confusion_matrix(mean,clases,axis=ax1,title=names[c]+" Mean")
    plot_confusion_matrix(std,clases,axis=ax2,title=names[c]+" Std")

outPath = os.path.join(folder,outName)

fig.savefig(outPath+'.png')
df.to_excel(outPath+'.xls', sheet_name='Sheet1', index=False, engine='xlsxwriter')
