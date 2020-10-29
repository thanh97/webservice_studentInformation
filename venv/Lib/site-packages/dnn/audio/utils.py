import librosa
import soundfile
import os
from rs4 import pathtool
from ..video.ffmpeg import run_process 

def resampling (audioFilePath, targetFilePath, sampling_rate = 16000):
    y, sr = librosa.load (audioFilePath, sr = 22050, mono = True)
    with open (targetFilePath, "wb") as f:
       soundfile.write (f, y, sampling_rate)
       
def resample_dir (directory, target = None, recusive = False):
    if taget is None:
        target  = os.path.join (directory, "AIMDV")
    pathtool.mkdir (target)
        
    for each in os.listdir (directory):
        if each == "AIMDV":
            continue
        
        path = os.path.join (directory, each)
        if os.path.isdir (path):
            if recusive:
                resample_dir (path, target, True)
            else:
                continue    
        
        try:
            resampling (path, target)
        except:
            raise

def convert2mp3 (input, output, quality = 7):
    try:
        int (quality)
    except ValueError:
        assert quality.endswith ("k")
        q = "-b:a {}".format (quality)
    else:
        q = "-q:a {}".format (quality)            
    os.path.isfile (output) and os.remove (output)
    cmd = 'ffmpeg -i "{}" -codec:a libmp3lame {} "{}"'.format (input, q, output)
    r = run_process (cmd)
    assert os.path.isfile (output)
    return output
    
def convert2wav (input, output, sample_rate = 44050):
    os.path.isfile (output) and os.remove (output)        
    cmd = 'ffmpeg -i "{}" -acodec pcm_s16le -ar {} "{}"'.format (input, sample_rate, output)
    r = run_process (cmd)
    #r = run_process ('ffmpeg -i "{}" "{}"'.format (input, output))
    assert os.path.isfile (output)
    return output
    
    