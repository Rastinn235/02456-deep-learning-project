import numpy as np
#pytorch parts
import torch
from torch.utils.data import DataLoader, Dataset
import torch.distributions as distributions
import torch.functional as F
#plotting
import matplotlib.pyplot as plt

#project specific
#import pandas as pd
import transformers as hf #hf = huggingFace

import dataload2 as dl
from settings import Settings
import midi
import model

# ------------------------


settings = Settings()
torch.backends.cudnn.enabled = False

printPlots = settings.printPlots
playSample = settings.playSample
forceCPU = settings.forceCPU
batchSize = settings.hyperparameters['batchSize']

print("hello \n")
useCuda,device = model.getDevice(forceCPU=forceCPU)
print("Cuda available: ",useCuda,' device:', device,'\n')

training,test,validation = dl.createMaestroDatasets(diminishedSet=settings.diminishedSet)

#Load 1. sample and check it
sampleInput,sampleTarget,samplefs = training[0]
print("printing shapes of a pianoRoll. Expected is [Nx128], as there is 128 midi notes, and length varies")
print("example shape: ", sampleInput.shape)

if(printPlots):
    #plot notes
    midi.plotPianoRoll(sampleInput,fs=samplefs)
    plt.figure()
    plt.show()
    midi.plotPianoRoll(sampleTarget,fs=samplefs)
    plt.figure()
    plt.show()
if(playSample):
    #play the piano roll
    midi.playPianoRoll(midiroll,playTime=2)

# create dataloaders
dataloaderTrain = DataLoader(training,batch_size=batchSize, shuffle=True)
dataloaderValidation = DataLoader(validation,batch_size=batchSize, shuffle=True)
dataloaderTest = DataLoader(test,batch_size=batchSize, shuffle=True)
# init net
net = model.LSTMnet(batchSize=batchSize)  #batchfirst [batch,seq,128]
print(net)
net = net.to(device) #Send to device (GPU or CPU)
# test net
sampleInputTensor = torch.Tensor(sampleInput).view(1,-1,128) #add batch dimension
sampleOutput = net(sampleInputTensor.to(device)) #send to network!

#permute back to [feature,seq], send to cpu, detach and convert to ndarray
sampleOutput = sampleOutput.cpu().detach().numpy()
if (printPlots):
    # plot notes
    midi.plotPianoRoll(sampleInput)
    plt.figure()
    plt.show()
    midi.plotPianoRoll(sampleOutput)
    plt.figure()
    plt.show()

del sampleInput,sampleInputTensor,sampleOutput,samplefs

print('Starting network training')
net = model.trainNetwork(net=net,trainSet=dataloaderTrain,testSet=dataloaderTest,validationSet=dataloaderValidation,cudaDevice=device)
torch.save(net,'net.pth')

print('Main finished!')
