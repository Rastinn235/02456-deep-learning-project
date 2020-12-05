import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import transformers as hf
import torch.distributions as distributions
from settings import Settings
import numpy as np
import matplotlib.pyplot as plt

class LSTMnet(nn.Module):
    def __init__(self,batchSize,dropoutProbability = -1,settings = Settings()):
        self.hiddenSize = settings.hyperparameters['LSTMHiddenSize']
        self.numLayers = settings.hyperparameters['LSTMLayers']
        if(dropoutProbability<0):
            dropoutProbability = settings.hyperparameters['dropoutProbability']

        super(LSTMnet,self).__init__()

        self.lstm = nn.LSTM(
                input_size=128,
                hidden_size=self.hiddenSize,
                num_layers=self.numLayers,
                bidirectional=False,
                batch_first=True,
                dropout=dropoutProbability)
        self.relu = nn.ReLU()

        self.l_out = nn.Linear(in_features=self.hiddenSize,
                               out_features=128,
                               bias=False)
        self.softmax = nn.Softmax(dim=1)


    def forward(self,x):
        x, (h,c) = self.lstm(x)

        # flatten for feed-forward layer
        x = x.view(-1,self.lstm.hidden_size)
        x = self.relu(x)
        x = self.l_out(x)
     #   x = self.softmax(x)
        return x

def getDevice(forceCPU = False):
    if torch.cuda.is_available() and not forceCPU:
        device = torch.device('cuda:0')
        useCuda = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

    else:
        device = torch.device('cpu') # don't have GPU
        useCuda = False
    return useCuda,device

def plotLosses(trainingLoss,validationLoss =[],testLoss = []):
    epoch = np.arange(len(trainingLoss))
    plt.figure()
    plt.plot(epoch, trainingLoss, 'r', label='Training loss')
    if(validationLoss):
        plt.plot(epoch, validationLoss, 'b', label='Validation loss')
    if(testLoss):
        plt.plot(epoch, validationLoss, 'b', label='Test loss')
    plt.legend()
    plt.xlabel('Epoch'), plt.ylabel('CEL')
    plt.show()

def trainNetwork(net,trainSet,testSet,validationSet,cudaDevice,settings=Settings()):
    numEpoch = settings.hyperparameters['epoch']
    lr = settings.hyperparameters['lr']
    weightDecay = settings.hyperparameters['weightDecay']
    batchSize = settings.hyperparameters['batchSize']
    printLossEverynEpoch = settings.printLossEverynEpoch
    if (settings.lowMemory):
        torch.cuda.empty_cache()

    criterion = nn.BCEWithLogitsLoss() #CrossEntropyLoss can not be used as we have multi-labeling (several notes at once)
    optimizer = optim.SGD(net.parameters(),lr = lr,weight_decay=weightDecay)

    trainingLoss,validationLoss = [],[]

    if(settings.lowMemory):
        #set to fp16 to reduce memory
        scaler = torch.cuda.amp.GradScaler()

    for epoch in range(numEpoch):
        print('trainNetwork: Starting Epoch {}'.format(epoch))
        epochTrainingLoss = 0
        epochValidationLoss = 0

        # training
        net.train()
        print('trainNetwork: model in training mode')
        idx = 0
        for input,target,_ in trainSet:
            print('  trainNetwork: training on sample {}'.format(idx))
            #input and target dim [batch,seq,feature]
            # convert inputs and targets to tensors and send to Cuda
            input = input.float()
            if(batchSize==1):
                input = input.view(1,-1,128) #some issues were seen where no batch dimension was added
            input = input.to(cudaDevice)

            target = target.float()

            if(batchSize==1):
                target = target.view(1,-1,128)
            target = target.to(cudaDevice)


            outputs = net(input)
            if(batchSize==1):
                outputs = outputs.view(1,-1,128)

            if(settings.lowMemory):
                with torch.cuda.amp.autocast():
                    loss = criterion(outputs,target)
            else:
                loss = criterion(outputs,target)
            #backward pass
            if(settings.lowMemory):
                torch.cuda.empty_cache()
            optimizer.zero_grad()
            if (settings.lowMemory):
                torch.cuda.empty_cache()
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if (settings.lowMemory):
               # print('After backward prop')
               # print('memory_allocated', torch.cuda.memory_allocated() / 1e9, 'memory_cached', torch.cuda.memory_cached() / 1e9)
                torch.cuda.empty_cache()
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            epochTrainingLoss += loss.detach().cpu().numpy()
            idx = idx+1
            if(settings.lowMemory):
                torch.cuda.empty_cache()


        # validation
        net.eval()
        print('trainNetwork: model in evaluation mode')
        idx = 0
        for input,target,_ in validationSet:
            print('  trainNetwork: validation on sample {}'.format(idx))
            input = input.float()
            if (batchSize == 1):
                input = input.view(1, -1, 128)  # some issues were seen where no batch dimension was added
            input = input.to(cudaDevice)
            target = target.float()
            target = target.to(cudaDevice)

            outputs = net(input)
            if (batchSize == 1):
                outputs = outputs.view(1, -1, 128)

            if(settings.lowMemory):
                with torch.cuda.amp.autocast():
                    loss = criterion(outputs,target)
            else:
                loss = criterion(outputs,target)
            epochValidationLoss += loss.detach().cpu().numpy()
            idx = idx+1
            if(settings.lowMemory):
                torch.cuda.empty_cache()

        trainingLoss.append(epochTrainingLoss/len(trainSet))
        validationLoss.append(epochTrainingLoss/len(validationSet))


        if epoch % printLossEverynEpoch == 0:
            print(f'Epoch {epoch}, training loss: {trainingLoss[-1]}, validation loss: {validationLoss[-1]}')

        plotLosses(trainingLoss=trainingLoss,validationLoss=validationLoss)

        #end epoch
    return net,trainingLoss,validationLoss
