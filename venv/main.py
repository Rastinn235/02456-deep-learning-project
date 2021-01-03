import numpy as np
#pytorch parts
import torch
from torch.utils.data import DataLoader, Dataset
import torch.distributions as distributions
import torch.functional as F
#plotting
import matplotlib.pyplot as plt
import os

#project specific
#import pandas as pd
import transformers as hf #hf = huggingFace
import dataload2 as dl
from settings import Settings
from Helperfunctions import sigmoid
import midi
import model

# ------------------------


settings = Settings()
torch.backends.cudnn.enabled = False

printPlots = settings.printPlots
playSample = settings.playSample
forceCPU = settings.forceCPU
batchSize = settings.hyperparameters['batchSize']
useSavedNet = settings.useSavedNet
useCheckpoints = settings.useCheckpoints

print("hello \n")
useCuda,device = model.getDevice(forceCPU=forceCPU)
print("Cuda available: ",useCuda,' device:', device,'\n')
if(settings.lowMemory):
    print("Using lowMemory settings")
    print("Autoscaling of loss and variables enabled")
training,test,validation = dl.createMaestroDatasets(diminishedSet=settings.diminishedSet)

#Load 1. sample and check it
sampleInput,sampleTarget,samplefs = training[0]
print("printing shapes of a pianoRoll. Expected is [Nx128], as there is 128 midi notes, and length varies")
print("example shape: ", sampleInput.shape)

if(printPlots):
    #plot notes
    midi.plotPianoRoll(sampleInput,fs=samplefs)
    plt.figure(num=1,figsize=(10,8),dpi=800)
    plt.show()
    midi.plotPianoRoll(sampleTarget,fs=samplefs)
    plt.figure(figsize=(10,8),dpi=800)
    plt.show()

if(playSample): #play sample pianoroll
    #play the piano roll
    midi.playPianoRoll(midiroll,fs=samplefs,playTime=2)

# create dataloaders
dataloaderTrain = DataLoader(training,batch_size=batchSize, shuffle=True)
dataloaderValidation = DataLoader(validation,batch_size=batchSize, shuffle=True)
dataloaderTest = DataLoader(test,batch_size=batchSize, shuffle=True)

# init net
if(useSavedNet):
    print('Using saved net: ',settings.savedNetPath)
    net = torch.load(settings.savedNetPath)
    net.eval()
else:
    print('Starting new network')
    net = model.LSTMnet(batchSize=batchSize)  #batchfirst [batch,seq,128]
if not useCheckpoints:
    if(os.path.exists(settings.checkpointPath)):
        print('Deleting previous checkpoints')
        os.remove(settings.checkpointPath)

print(net)
net = net.to(device) #Send to device (GPU or CPU)
# test net
sampleInputTensor = torch.Tensor(sampleInput).view(1,-1,128) #add batch dimension
sampleOutput = net(sampleInputTensor.to(device)) #send to network!
sampleOutput = sampleOutput.squeeze() #remove additional dimension
#send to cpu, detach and convert to ndarray
sampleOutput = sigmoid.sigmoid(sampleOutput.cpu().detach().numpy())

if(settings.playSample):
    print('playing sample output')
    midi.playPianoRoll(sampleOutput, fs=samplefs,)
if (printPlots):
    # plot notes
    midi.plotPianoRoll(sampleInput,fs=samplefs)
    plt.figure()
    plt.show()
    midi.plotPianoRoll(sampleOutput,fs=samplefs)
    plt.figure()
    plt.show()

del sampleInput,sampleInputTensor,sampleOutput,samplefs

print('Starting network training')
net = model.trainNetwork(net=net,trainSet=dataloaderTrain,testSet=dataloaderTest,validationSet=dataloaderValidation,cudaDevice=device)

torch.save(net[0],settings.savedNetPath)

print('Main finished!')
