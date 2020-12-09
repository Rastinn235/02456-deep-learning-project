from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from mido import MidiFile
from settings import Settings
import os
from tqdm import tqdm


if __name__ == '__main__':
    settings = Settings()
    dataset = MaestroDataset(settings = settings)
    pianoRoll, velocity = dataset[0]

class Dataload(Dataset):
    def __init__(self,settings:Settings,datasetName: str, preprocess=False):
        self.settings = settings

        if datasetName in self.settings.dataset_info:
            self.datasetPath = s.path.join(self.settings.datasetRoot, self.settings.dataset_info[datasetName]['path'])
        else:
            raise Exception('dataload: Dataset key does not exist in settings')

        #check if dataset exists
        if not os.path.exists(self.datasetPath):
            raise Exception('dataload: Dataset path does not exists')

        if(preprocess):
            raise NotImplementedError
#------------------------------


class MaestroDataset(Dataload):
    def __init__(self, splitmode, settings:Settings):
        datasetName = 'Maestro'
        super().__init__(settings=settings, datasetName = datasetName)

        self.chunk = self.ticks2chunk(metaTicks = self.getMetaticks())[key]


#to be able to index in dataset
    def __getitem__(self, index):
        assert index >= 0 and index<self.__len__(), "Dataset index out of bounds. Must be between 0 and len"


#helper functions
    def __len__(self):
        return self.chunk[-1]["length"]

    def getRoot(self):
        return os.path.join(self.datasetPath, "period_{}".format(self.settings.quantization_period))

    def getLocalIndex(self,index):
        fileIndex = search(self.chunk,index)
        fileName = self.chunk[fileIndex]["name"]


        if fileIndex !=0:
            localIndex = index - self.chunk[fileIndex-1]["length"]
        else:
            localIndex = index

        return fileIndex,fileName,localIndex

    def ticks2chunk(self,ticks):
        chunk = {}
        for key in ticks.keys():
            chunk[key] = []

        for key in ticks.keys():
            chunkSum = 0
            for file in ticks[key]:
                chunkSum += int(file["length"] / self.settings.length)
                tmp = {}
                tmp["name"] = file["name"]
                tmp["length"] = chunkSum
                chunk[key].append(tmp)
        return chunk
#----------------------------------------------------------------------

    def loadData(batchsize,csvPath):
        csvread = pd.read_csv(csvPath)
        print(list(csvread.columns))

        n = len(csvread)
        split = csvread.iloc[0:n - 1, 2]
        print('training set size: ', np.sum(split == 'train'))
        print('Test set size: ', np.sum(split == 'test'))
        print('validation set size: ', np.sum(split == 'validation'))

        train = csvread[csvread['split'].str.match('train')]
        test = csvread[csvread['split'].str.match('test')]
        validation = csvread[csvread['split'].str.match('validation')]

        # 'midi_filename' is column with paths to midi data
        print(train['midi_filename'])


        trainLoad = DataLoader(train, batch_size=batchsize, shuffle=True)
        testLoad = DataLoader(test, batch_size=batchsize, shuffle=True)
        validationLoad = DataLoader(validation, batch_size=batchsize, shuffle=True)
        print('dataload: loading done')

        return trainLoad, testLoad, validationLoad