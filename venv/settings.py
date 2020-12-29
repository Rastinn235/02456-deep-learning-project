import os


class Settings:
    root = os.path.dirname(os.path.realpath(__file__))

    # Dataset Settings
    datasetRoot = os.path.join(root, '/Dataset')
    dataset_info = {
        'Maestro': {
            'path': './maestro-v2.0.0' #,
        }
    }

    hyperparameters = {
        'epoch': 10,
        'lr': 0.1,
        'weightDecay': 0,
        'batchSize':5,
        'LSTMHiddenSize': 100,
        'LSTMLayers': 3,
        'dropoutProbability': 0.3
    }
    # various settings
    useSavedNet = False
    savedNetPath = 'net.pt'
    pianoThresholding = 0.1 #other tests: 0.1,0.01 #1/128 = random
    diminishedSet = False
    diminishFactor = 100
    printLossEverynEpoch = 1
    playSample = False
    printPlots = True
    lowMemory = True #try to reduce memory requirements through clearing caches and using fp16 for all gradients, etc
    reduceSequenceLengthSeconds = 60 # reduce sequence length to n seconds. Set to a value between [-inf:0] to have full sequence lengths
                                        # will also zero-pad if sequencelength < reduceSequenceLengthSeconds
    forceCPU = False #force to work on CPU instead of autoselect device based on found CUDAs


