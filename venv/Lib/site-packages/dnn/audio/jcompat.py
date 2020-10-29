import numpy as np
import math

def mfcc (melspec, n_mfccs):
  # mfcc for android  
  nmel, frames = melspec.shape        
  
  m = math.sqrt(2.0 / nmel)
  DCTcoeffs = np.zeros((nmel, n_mfccs))
  
  a = np.arange (0.5, nmel, 1.0).reshape ([-1, 1])
  b = np.arange (1, n_mfccs + 1, 1.0).reshape ([1, -1])
  DCTcoeffs = a.dot (b)
  DCTcoeffs = m * np.cos (DCTcoeffs * math.pi / nmel)
    
  mfccs = np.zeros ((frames, n_mfccs))
  DCTcoeffs_ = DCTcoeffs.swapaxes (1, 0)
  
  # convert from dB to plain old log magnitude
  melspec_ = melspec / 10.
  for frm in range (frames):
      mfccs [frm] +=  DCTcoeffs_.dot (melspec_ [:,frm]) / frames
  return mfccs.swapaxes(1, 0);
