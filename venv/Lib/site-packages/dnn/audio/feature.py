import numpy as np
import sys
import os
import glob
import wave
import multiprocessing
from functools import partial
from scipy.fftpack import fft
from scipy.io import wavfile
from scipy import signal
from tqdm import tqdm
import librosa
import librosa.util
import random 
from scipy.stats import skew, kurtosis
import warnings
import resampy
from . import augments

SAMPLE_RATE = 16000

def _save (data, wavfile, target_path):
    savedir = os.path.join (target_path, os.path.basename (os.path.dirname (os.path.dirname (wavfile))))
    if not os.path.isdir (savedir):
        os.mkdir (savedir)
    target_file = os.path.join(savedir, "%s.npy" % os.path.basename (wavfile))
    np.save(target_file, data)

def get_volume (y):
    return np.sqrt (np.sum (y ** 2))

def norm_volmue (y):
    return y / np.max (np.abs (y))
_normalize = norm_volmue

def whitening (y):
    mean = np.mean (y)    
    std = np.std (y)  
    return (y - mean) / std
    
def log (x, name = ""):
    return np.log (np.abs (x) + 1e-3) / 6.7
  
def add_padding (x, padding):
    if not padding:
        return x
    return np.concatenate ([x, np.zeros (padding)])

def to_db (S):
  S = np.abs (S) + 1e-10 
  return 10 * np.log10(S) - 10 * np.log10(np.max (S))

def gain (input, output, target_dBFS = -20.0):
    from pydub import AudioSegment
    
    sound = AudioSegment.from_file (input)
    change_in_dBFS = target_dBFS - sound.dBFS
    normalized_sound = sound.apply_gain (change_in_dBFS)
    normalized_sound.export(output, output.split (".")[-1].lower ())
    return normalized_sound.dBFS
        
def _compress (data, name, padding = 0, valid_max = 2.0, valid_min = -2.0):
    # padding 3 means 2 convolution, 1 polling    
    data = data.real
    if name == "poly_features":
        data = data / 10.
    elif name == "rmse":
        data = data / 4.
    elif name == "spectral_contrast":
        data = data / 50.
    elif name == "mel":
        data = data / -80.
    elif name in ("stft", "spectral_centroid", "spectral_bandwidth", "spectral_rolloff"):        
        data = log (data, name)
                          
    features = np.array ([
        add_padding (np.mean (data, axis = 1), padding),
        add_padding (np.max (data, axis = 1), padding),
        add_padding (np.min (data, axis = 1), padding),
        add_padding (np.median (data, axis = 1), padding),
        add_padding (np.var (data, axis = 1), padding),
        add_padding (skew (data, axis = 1) / 15., padding),
        add_padding (np.log (abs (kurtosis (data, axis = 1)) + 1e-3) / 6.7, padding)
    ])
    if np.min (features) < valid_min or np.max (features) > valid_max:
        #print (name, int (np.max (features)), int (np.max (features)))
        raise AssertionError
    return features

def augment (y, sr, time_stretch = False, pitch_shift = False, random_laps = False, add_noise = False):    
    if add_noise:
        y = augments.add_noise (y, sr, 0.01, 0.12)            
    if time_stretch:
        y = augments.speed (y, sr, 0.1)
    if pitch_shift:
        y = augments.pitch_shift (y, sr, 0.5)    
    if random_laps and librosa.get_duration(y, sr = sr) > 1.6:
       y = augments.random_laps (y, sr)
    return y   

def augment_with_norm (y, sr = SAMPLE_RATE, *args, **karg):
    y = augment (y, sr, *args, **karg)
    y = norm_volmue (y)
    return whitening (y)

def loadraw (wavfile, sample_rate = SAMPLE_RATE, time_lap = 0.2, due_limit = (0.5, 10), trim_db = 60, res_type = "kaiser_best"):
    if res_type == "hann_window":
        y, sr_orig = librosa.load (wavfile, None, True)
        y = resampy.resample(y, sr_orig, sample_rate, filter = 'sinc_window', window = signal.hann)
        sr = sample_rate
    else:
        y, sr = librosa.load (wavfile, sample_rate, True, res_type = res_type)
    
    if time_lap:
        removable = int (sample_rate * time_lap)
        y = y [removable:len (y) - removable]
    # Trim the beginning and ending silence
    if trim_db:
        try:
            y, index = librosa.effects.trim (y, trim_db)
        except ValueError:        
            return None, sr   
    if due_limit:    
        duration = librosa.get_duration(y, sr = sr)
        #print ("    + duration: {:2.1f} sec => {:2.1f} sec".format (a_duration, duration))
        if duration < due_limit [0] or duration > due_limit [1]:
            warnings.warn ("audio file length is invalid, ignored")
            return None, sr
    if len (y) % 2 == 1:
        y = y [:-1]
    return y, sr

def load (wavfile, sample_rate = SAMPLE_RATE, res_type = "kaiser_best"):
    return loadraw (wavfile, sample_rate, None, None, 0, res_type)
    

if __name__ == '__main__':    
    pass
