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
        'epoch': 8,
        'lr': 0.5,
        'weightDecay': 0.00,
        'batchSize':1,
        'LSTMHiddenSize': 100,
        'LSTMLayers': 4,
        'dropoutProbability': 0.3
    }
    # various settings
    useSavedNet = False
    useCheckpoints = False
    savedNetPath = "net.pt"
    checkpointPath = "checkpoint.pt"
    pianoThresholding = 0.1 #other tests: 0.1,0.01 #1/128 = random
    diminishedSet = False
    diminishFactor = 10
    printLossEverynEpoch = 1
    playSample = False
    printPlots = True
    lowMemory = True #try to reduce memory requirements through clearing caches and using fp16 for all gradients, etc
    reduceSequenceLengthSeconds = 60 # reduce sequence length to n seconds. Set to a value between [-inf:0] to have full sequence lengths
                                        # will also zero-pad if sequencelength < reduceSequenceLengthSeconds
    forceCPU = False #force to work on CPU instead of autoselect device based on found CUDAs


