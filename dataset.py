import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split


def labels_to_one_hot(labels):
    ''' Converts list of integers to numpy 2D array with one-hot encoding'''
    N = len(labels)
    one_hot_labels = np.zeros([N, 2], dtype=int)
    one_hot_labels[np.arange(N), labels] = 1
    return one_hot_labels

#Todo crear indice y dar batch en demand nomas
class Dataset:
    def __init__(self,dataFolder, batch_size=100,testProp=0.3, validation_proportion=0.1):
        # Training set 181 datos
        # 30 Clase 1 Train, 30 Clase 1 Test. 60 Clase 0 Train, 60 clase 0 Test
        #140      41
        labels = []
        data = []

        seed = 1 #THIS MAKE THAT THE TRAIN VAL TEST REMAIN THE SAME IF YOU RE RUN (random number generator seed)
        np.random.seed(seed=seed)


        #Here open files in folder somehow
        tlabels = [[],[]]
        all = []
        allL = []
        for ind,f in enumerate(os.listdir(dataFolder)):
            ray_image = cv2.cvtColor(cv2.imread(os.path.join(dataFolder,f)), cv2.COLOR_BGR2GRAY)
            label = int(f.split("_")[1][0])
            tlabels[label].append(ind)
            all.append(ray_image)
            allL.append(label)

        orderA = np.random.permutation(60)
        orderB = np.random.permutation(120)
        tlabels[0] = np.array(tlabels[0])
        tlabels[1] = np.array(tlabels[1])
        all = np.array(all)
        allL = np.array(allL)


        tvdata = all[np.array(list(tlabels[0][orderA[:30]]) + list(tlabels[1][orderB[:60]]))]
        tv_labs = np.array([0 for i in range(30)] + [1 for i in range(60)])
        sf = np.random.permutation(90)
        tvdata = tvdata[sf]
        tv_labs = tv_labs[sf]

        test_data = all[np.array(list(tlabels[0][orderA[30:]]) + list(tlabels[1][orderB[60:]])) ]
        test_labels = np.array([0 for i in range(30)] + [1 for i in range(60)])
        sf = np.random.permutation(90)
        test_data = test_data[sf]
        test_labels = test_labels[sf]

        #separar trainVal -  test
        # tvdata,test_data,tv_labs,test_labels = train_test_split(data,labels, test_size=testProp,random_state=seed)

        #separar trainVal en val - train
        assert validation_proportion > 0. and validation_proportion < 1.
        tdata, vdata, tlabels, vlabels = train_test_split(tvdata, tv_labs, test_size=validation_proportion, random_state=seed)

        self.train_data = tdata
        self.train_labels = labels_to_one_hot(tlabels)
        self.validation_data = vdata
        self.validation_labels = labels_to_one_hot(vlabels)
        self.test_data = test_data
        self.test_labels = labels_to_one_hot(test_labels)


        # Normalize data
        self.mean = self.train_data.mean(axis=0)
        self.std = self.train_data.std(axis=0)
        self.train_data = np.array([   (im-self.mean)/self.std for im in self.train_data])
        self.validation_data = np.array([   (im-self.mean)/self.std for im in self.validation_data])
        self.test_data =  np.array([   (im-self.mean)/self.std for im in self.test_data])



        # Augment training dataset (horizontal flipping)
        # np.random.seed(seed=1)
        # if augment_data:
        #     flipped_train_data = self.train_data[:, :, ::-1, :]
        #     self.train_data = np.concatenate([self.train_data, flipped_train_data],
        #                                      axis=0)
        #     self.train_labels = np.concatenate([self.train_labels, self.train_labels],
        #                                        axis=0)
        #
        #     # shuffle training set
        #     new_idx = np.random.permutation(np.arange(len(self.train_labels)))
        #     self.train_data = self.train_data[new_idx]
        #     self.train_labels = self.train_labels[new_idx]

        # Batching & epochs
        self.batch_size = batch_size
        self.n_batches = len(self.train_labels) // self.batch_size
        self.current_batch = 0
        self.current_epoch = 0

    def normalizeImage(self,image):
        return (image-self.mean)/self.std

    def nextBatch(self):
        ''' Returns a tuple with batch and batch index '''
        start_idx = self.current_batch * self.batch_size
        end_idx = start_idx + self.batch_size
        batch_data = self.train_data[start_idx:end_idx].reshape((20,96,96,1))
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
                nElem = min(20,data.shape[0])
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
    cifar10 = Dataset("data/mix",batch_size=20)
