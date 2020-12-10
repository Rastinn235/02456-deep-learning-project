# plot random piano roll
midiroll = midi.midi2roll(listofAllMidiFiles[1])
midi.plotPianoRoll(midiroll)
plt.figure(figsize=(8, 4))
plt.show()
train, target = midi.splitLowHighNotes(midiroll)
print("split")
midi.plotPianoRoll(train)
plt.figure(figsize=(8, 4))
plt.show()
midi.plotPianoRoll(target)
plt.figure(figsize=(8, 4))
plt.show()

# test input/target creation
tmp = np.zeros((50,128))
tmp[25,100] = 1
tmpInput,tmpTarget = dl.createInputsTargetsFromPianoRoll(tmp)
assert((tmpInput[25,:] == tmpTarget[24,:]).all())

"""
class Dataload:
    def __init__(self, settings: Settings, datasetName: str,):
        self.settings = settings

        if datasetName in self.settings.dataset_info:
            self.datasetPath = s.path.join(self.settings.datasetRoot, self.settings.dataset_info[datasetName]['path'])
        else:
            raise Exception('dataload: Dataset key does not exist in settings')

        # check if dataset exists
        if not os.path.exists(self.datasetPath):
            raise Exception('dataload: Dataset path does not exists')




class MaestroDataload(Dataload):
    def __init__(self,settings:Settings):
        datasetName = 'Maestro'
        super().__init__(settings=settings,datasetName=datasetName)

        csvread = pd.read_csv(settings.dataset_info[datasetName].get(path))

"""

#  for i_batch, sample_batched in enumerate(dataloader):
#      print(i_batch,sample_batched.size())


def validation():
    raise NotImplementedError
    torch.distributions.Normal(self.mu,self.sigma)
    # implement validation metrics: (log p(x) = \log p(x_0) + \sum_i log p(x_i | x_{<i}))
