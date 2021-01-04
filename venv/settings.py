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
        'epoch': 30,
        'lr': 0.9,
        'weightDecay': 0,
        'batchSize':1,
        'LSTMHiddenSize': 100,
        'LSTMLayers': 3,
        'dropoutProbability': 0.5
    }
    # various settings
    useSavedNet = True
    useCheckpoints = True
    savedNetPath = "net.pt"
    checkpointPath = "checkpoint.pt"

    pianoThresholding = 0.0024
    diminishedSet = False
    diminishFactor = 10
    printLossEverynEpoch = 1
    playSample = True
    playSampleTime = 2
    printPlots = True
    lowMemory = True #try to reduce memory requirements through clearing caches and using fp16 for all gradients, etc
    reduceSequenceLengthSeconds = 60 # reduce sequence length to n seconds. Set to a value between [-inf:0] to have full sequence lengths
                                        # will also zero-pad if sequencelength < reduceSequenceLengthSeconds
    forceCPU = False #force to work on CPU instead of autoselect device based on found CUDAs


