import pretty_midi
from settings import Settings
#import pandas as pd
import numpy as np

import librosa.display
import os
import matplotlib as plt
from Helperfunctions import reversepianoRoll
from sounddevice import play,stop
import time
import warnings

defaultFs = 50 #100 #frequency of column updates per second. i.e each column is spaced apart by 1/fs seconds
    # should probably be made to depend on the tempo. So defaultFS = tempo*12; 12 times per quarter note allows to have proper timing
    # of both quarter and triplet notes.
audioFs = 16000 #audio sample rate frequency. Used in playPianoRoll

def midi2roll(midiFileName,instrument = 0,fs=defaultFs,settings = Settings()):
    """
    :param midiFileName: filename including path of the midi file
    :param instrument:  which instrument to extract to roll. http://fmslogo.sourceforge.net/manual/midi-instrument.html
    :param fs: frequency of column updates. i.e each column is spaced apart by 1/fs seconds
    :return:
    """
    midiPretty = pretty_midi.PrettyMIDI(midiFileName)

    _,midiTempo = midiPretty.get_tempo_changes() #get tempo events
    if(midiTempo.size> 0):
        #midiTempo is in BPM
        fs = int(np.ceil(midiTempo[0]/60 *12)) #bpm/60 = bps, *12 to get both 16th and triols

    midiPiano = midiPretty.instruments[instrument] #channel 0 = piano channel if there are multiple
    pianoRoll = midiPiano.get_piano_roll(fs)
    if(settings.reduceSequenceLengthSeconds>0):
        idx = int(np.round(fs * settings.reduceSequenceLengthSeconds))
        if(len(pianoRoll[1])>idx): #only reduce length if song is longer than idx
            pianoRoll = pianoRoll[:,0:idx]
        else: #else zero-pad
            pianoRoll = np.hstack((pianoRoll,np.zeros((128,idx-len(pianoRoll[1])))))
    return pianoRoll, fs

def splitLowHighNotes(pianoRoll,noteSplit = 53):
    """

    :param pianoRoll: input piano roll to be split [L x 128]
    :param noteSplit: note to between high and low notes. Notesplit will be part of high notes
    :param settings: settings
    :return lowNotes: all notes below noteSplit [L x 128]
    :return highNotes: all notes at or above noteSplit [L x 128]

    """
    lowNotes = np.zeros(pianoRoll.shape)
    highNotes = np.zeros(pianoRoll.shape)

    lowNotes[:,0:noteSplit] = pianoRoll[:,0:noteSplit] #all notes below noteSplit
    highNotes[:,noteSplit:] = pianoRoll[:,noteSplit:] #all notes above noteSplit

    return lowNotes,highNotes

def joinLowHighNotes(lowNotes,highNotes,noteSplit = 53):
    """
    :param lowNotes: all notes below noteSplit [L x 128]
    :param highNotes: all notes at or above noteSplit [L x 128]
    :param
    :param noteSplit: note to between high and low notes. Notesplit will be part of high notes [int between 0-127]
    :param settings: settings
    :return pianoRoll: joined low and high notes to piano roll [L x 128]
    """
    pianoRoll = np.zeros(lowNotes.shape)
    pianoRoll[:,0:noteSplit] = lowNotes[:,0:noteSplit]
    pianoRoll[:,noteSplit:] = highNotes[:,noteSplit:]
    return pianoRoll

def thresholdPianoRoll(roll, threshold = 0.5):
    """
    :param roll: PianoRoll [L x 128]
    :param threshold: threshold to remove noise [0-1]
    :return:
    """
    thresholdedRoll = (roll >= threshold) * 1
    return thresholdedRoll

def plotPianoRoll(roll, startPitch=21,endPitch=108,settings = Settings(),fs = defaultFs):
    """
    :param roll: pianoroll [Lx128]
    :param startPitch: start pitch in midi note [1-128] (21 = 1. possible piano note on 88 key grand)
    :param endPitch: end pitch in midi note [1 128] (108 = last possible piano note on 88 key grand)
    :param fs: frequency of column updates. i.e each column is spaced apart by 1/fs seconds
    :return:

    """
    threshold = settings.pianoThresholding
    roll = thresholdPianoRoll(roll,threshold=threshold)
    roll = roll.transpose()
    # Use librosa's specshow function for displaying the piano roll
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        librosa.display.specshow(roll[startPitch:endPitch],
             hop_length=1, sr=fs, x_axis='time', y_axis='cqt_note',
             fmin=pretty_midi.note_number_to_hz(startPitch))



def playPianoRoll(instrumentRoll,sf2Path=None,audioFs = audioFs,fs=defaultFs,playTime = -1,settings=Settings()):
    """

    :param instrumentRoll: pianoroll [L x 128]
    :param sf2Path: path to sf2 file if using fluidsynth
    :param audioFs: created audio sample rate
    :param fs: piano roll sample rate
    :param playTime: how long to play
    :return:
    """
    threshold = settings.pianoThresholding
    instrumentRoll = thresholdPianoRoll(instrumentRoll,threshold=threshold)
    instrumentRoll = instrumentRoll.transpose()

    instrument = reversepianoRoll.piano_roll_to_pretty_midi(instrumentRoll,fs)

    try:
        if os.path.exists(sf2Path):
            audio = instrument.fluidsynth(fs=audioFs,sf2_path=sf2Path)
        else:
            print('midi:playPianoRoll: no sf2 file at given path')
    except:
        print('midi:playPianoRoll: no Fluidsynth installed. Synthesizing using sine waves')
        audio = instrument.synthesize(fs=audioFs)

    if (playTime<0):
        playTime = (len(audio) / audioFs)

    play(audio,audioFs)
    time.sleep(playTime)
    stop()
