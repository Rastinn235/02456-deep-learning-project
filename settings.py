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
        'epoch': 500,
        'lr': 0.1,
        'weightDecay': 0,
        'batchSize':1,
        'LSTMHiddenSize': 50,
        'LSTMLayers': 1,
        'dropoutProbability': 0
    }
    # various settings
    pianoThresholding = 0.5
    diminishedSet = False
    diminishFactor = 100
    printLossEverynEpoch = 1
    playSample = False
    printPlots = False
    lowMemory = False #try to reduce memory requirements through clearing caches
    forceCPU = False #force to work on CPU instead of auto

    quantizationPeriod = 16
    length = 128 # 4 bars in 1/16 period
