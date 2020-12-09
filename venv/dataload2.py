

from torch.utils.data import DataLoader, Dataset
import torch
import pandas as pd
import numpy as np
from settings import Settings
import os
import pretty_midi
import glob
import random

import midi

def getListofMidiFiles(settings = Settings(),folder = 'Dataset/maestro-v2.0.0/**/',nExpected=-1):
    """ Get a list of all midi files in the specified folder
        Parameters:
            folder: str
                midi folder inside datasetRoot folder set in settings.py. All subfolders are also included
            settings: function
                Settings file which sets up datasetRoot and other setup

        Returns:
            shuffled list of all midi files
    """
    #folder = os.path.join(settings.datasetRoot,folder) # TODO: make settings work<
    listAll = glob.glob(os.path.join(folder,'*.midi'))
   # random.shuffle(listAll) #no need to shuffle in the list. This is done by the dataloader
    if listAll == []:
        raise UserWarning('dataload:getListofMidiFiles: no MIDI files found')
    else:
        print('Searched for midi files in: ',folder)
        if(nExpected>0):
            print('Found ',len(listAll),' of ', nExpected,' midi files')
        else:
            print('Total midi files found:',len(listAll))
    return listAll

def createInputsTargetsFromPianoRoll(pianoRoll):
    """
    :param pianoRoll: sequence of notes
    :return: inputs and targets, where targets are same as inputs but shifted by 1
    """

    inputs = pianoRoll
    targets = np.roll(pianoRoll,shift = 1,axis = 0)
    return inputs, targets

class MaestroDataset(Dataset):
    def __init__(self,set):
        self.set = set
    def __len__(self):
        return len(self.set)
    def __getitem__(self, idx):
        tmp, fs = midi.midi2roll(self.set[idx])
        tmp = tmp.transpose() #transpose so it is [seq,feature]
        inputs, targets = createInputsTargetsFromPianoRoll(tmp)
        return inputs, targets, fs

def createMaestroDatasets(datasetFolder = 'Dataset/maestro-v2.0.0/',transform=None,diminishedSet = False,settings=Settings()):

    diminishFactor = settings.diminishFactor
    csvread = pd.read_csv(datasetFolder + 'maestro-v2.0.0.csv')
    if(len(csvread)>0):
        print('createMaestroDatasets: Found CSV file')
    else:
        print('createMaestroDatasets: No CSV file found')
    split = csvread.iloc[:, 2]
    midipath = csvread.iloc[:, 4]

    allFoundMidiFiles = getListofMidiFiles(folder=datasetFolder + '**/', nExpected=len(csvread))
    listOfMidiFiles = []

    validationSet = []
    trainingSet = []
    testSet = []

    # run through each of the sets
    for ii in range(len(csvread)):
        if (split[ii] == 'train'):
            trainingSet.append(datasetFolder + midipath[ii])
        elif (split[ii] == 'test'):
            testSet.append(datasetFolder + midipath[ii])
        elif (split[ii] == 'validation'):
            validationSet.append(datasetFolder + midipath[ii])

    print('{} midi files found with split \'{}\''.format(len(trainingSet), 'train'))
    print('{} midi files found with split \'{}\''.format(len(testSet), 'test'))
    print('{} midi files found with split \'{}\''.format(len(validationSet), 'validation'))

    #diminish dataset
    if(diminishedSet ==True):
        print('Diminished dataset enabled with factor {}'.format(diminishFactor))

        #ceil to always have at least 1 sample in each set
        trainingSet = trainingSet[0:int(np.ceil(len(trainingSet)/diminishFactor))]
        testSet = testSet[0:int(np.ceil(len(testSet)/diminishFactor))]
        validationSet = validationSet[0:int(np.ceil(len(validationSet)/diminishFactor))]
        print('Diminished {}set with {} first midi files created'.format('train',len(trainingSet)) )
        print('Diminished {}set with {} first midi files created'.format('test',len(testSet)) )
        print('Diminished {}set with {} first midi files created \n'.format('validation', len(validationSet)))

    trainingDataset = MaestroDataset(trainingSet)
    testDataset = MaestroDataset(testSet)
    validationDataset = MaestroDataset(validationSet)

    #returns 3 maestroDataset objects
    return trainingDataset,testDataset,validationDataset