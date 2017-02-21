import os
import numpy as np
from skimage.color import rgb2gray
import skimage.io as io
from sklearn.model_selection import train_test_split

def grayscaleEq(rgbimage):
    cof = 1.0/3
    return rgbimage[:,:,0] * cof + rgbimage[:,:,1] * cof + rgbimage[:,:,2] * cof
def labels_to_one_hot(labels,n):
    ''' Converts list of integers to numpy 2D array with one-hot encoding'''
    N = len(labels)
    one_hot_labels = np.zeros([N, n], dtype=int)
    one_hot_labels[np.arange(N), labels] = 1
    return one_hot_labels

#Todo crear indice y dar batch en demand nomas
class Dataset:
    def __init__(self,dataFolder, batch_size=100,testProp=0.3, validation_proportion=0.3,seed = 1):
        # Training set 181 datos
        # 30 Clase 1 Train, 30 Clase 1 Test. 60 Clase 0 Train, 60 clase 0 Test
        #140      41
        self.classes = 0
        self.fileNames = []
        labels = []
        data = []

        #THIS MAKE THAT THE TRAIN VAL TEST REMAIN THE SAME IF YOU RE RUN (random number generator seed)
        np.random.seed(seed=seed)


        #Here open files in folder somehow
        all = []
        allL = []
        fileList = enumerate(os.listdir(dataFolder))

        supImage = ["jpg", 'png']
        #Read all the images and labels
        for ind,f in fileList:
            if f.split('.')[-1] in supImage:
                ray_image = grayscaleEq(io.imread(os.path.join(dataFolder,f)))
                label = int(f.split("_")[1][0])
                all.append(ray_image)
                allL.append(label)
                self.fileNames.append(f)
        self.classes = len(set(allL)) #Get how many different classes are in the problem
        all = np.array(all)
        allL = np.array(allL)
        indAll = np.arange(all.shape[0]) #Get index to data to asociate filenames with data

        #split trainVal -  test
        tvdata,test_data,tv_labs,test_labels,indTrainVal,indTest = train_test_split(all,allL,indAll, test_size=testProp,random_state=seed)

        #separar trainVal en val - train
        assert validation_proportion > 0. and validation_proportion < 1.
        tdata, vdata, tlabels, vlabels,indTrain,indVal = train_test_split(tvdata, tv_labs,indTrainVal, test_size=validation_proportion, random_state=seed)

        self.train_data = tdata
        self.train_labels = labels_to_one_hot(tlabels,self.classes)
        self.validation_data = vdata
        self.validation_labels = labels_to_one_hot(vlabels,self.classes)
        self.test_data = test_data
        self.test_labels = labels_to_one_hot(test_labels,self.classes)

        self.trainInd = indTrain
        self.valInd = indVal
        self.testInd = indTest


        # Normalize data
        self.mean = self.train_data.mean(axis=0)
        self.std = self.train_data.std(axis=0)
        self.train_data = np.array([   (im-self.mean)/self.std for im in self.train_data])
        self.validation_data = np.array([   (im-self.mean)/self.std for im in self.validation_data])
        self.test_data =  np.array([   (im-self.mean)/self.std for im in self.test_data])

        #If we find image with no variation std = 0 and the data will have mean / 0 = nan. Replace nan with zero
        self.train_data = np.nan_to_num(self.train_data)
        self.validation_data = np.nan_to_num(self.validation_data)
        self.test_data = np.nan_to_num(self.test_data)

        # Batching & epochs
        self.batch_size = batch_size
        self.n_batches = len(self.train_labels) // self.batch_size
        self.current_batch = 0
        self.current_epoch = 0

    def getTrainFilename(self,trainIndex):
        return self.fileNames[self.trainInd[trainIndex]]

    def getValFilename(self,valIndex):
        return self.fileNames[self.valInd[valIndex]]

    def getTestFilename(self,testIndex):
        return self.fileNames[self.testInd[testIndex]]

    def getNclasses(self):
        return self.classes

    def normalizeImage(self,image):
        val = (image-self.mean)/self.std
        return np.nan_to_num(val)

    def nextBatch(self):
        ''' Returns a tuple with batch and batch index '''
        start_idx = self.current_batch * self.batch_size
        end_idx = start_idx + self.batch_size
        batch_data = self.train_data[start_idx:end_idx].reshape((self.batch_size,96,96,1))
        batch_labels = self.train_labels[start_idx:end_idx]
        batch_idx = self.current_batch

        # Update self.current_batch and self.current_epoch
        self.current_batch = (self.current_batch + 1) % self.n_batches
        if self.current_batch != batch_idx + 1:
            self.current_epoch += 1

        return ((batch_data, batch_labels), batch_idx)

    def getEpoch(self):
        return self.current_epoch


    def getSample(self,data,labels, asBatches = False):
        if asBatches:
            batches = []
            for i in range(max(1,len(data) // self.batch_size)):
                start_idx = i * self.batch_size
                end_idx = start_idx + self.batch_size
                nElem = min(self.batch_size,data.shape[0])
                batch_data = data[start_idx:end_idx].reshape((nElem,96,96,1))
                batch_labels = labels[start_idx:end_idx]
                batches.append((batch_data, batch_labels))
            return batches
        else:
            return (data, labels)

    def getValidationSet(self, asBatches=False):
        return self.getSample(self.validation_data,self.validation_labels,asBatches)
    def getTestSet(self, asBatches=False):
        return self.getSample(self.test_data, self.test_labels, asBatches)


    def reset(self):
        self.current_batch = 0
        self.current_epoch = 0


if __name__ == '__main__':
    cifar10 = Dataset("data/BaselineOuterInner",batch_size=20)
