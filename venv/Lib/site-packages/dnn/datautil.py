import numpy as np
from sklearn.model_selection import train_test_split
import random
import os
from colorama import Fore
from tqdm import tqdm

CHUNK_SIZE = 10000
def flatten (xs, ys, origin_index = None):    
    assert  len (xs) == len (ys)
    _xs = []
    _ys = []
    for segment in range (0, len (xs), CHUNK_SIZE):
        chunks = segment + CHUNK_SIZE
        small_xs, xs = xs [:chunks], xs [chunks:]
        small_ys, ys = ys [:chunks], ys [chunks:]
        for i, x_list in enumerate (small_xs):
            if origin_index is not None:
                x_list = [x_list [origin_index]] # use original only
            for x in x_list:           
                _xs.append (x)
                _ys.append (small_ys [i])
    return np.array (_xs), np.array (_ys)
_flatten = flatten

def get_tqdm (total, desc = "", bar_color = "white"):
    return tqdm (total = total, desc = desc, bar_format = "{l_bar}%s{bar}%s{r_bar}" % (getattr (Fore, bar_color.upper ()), Fore.RESET))

def get_size (size, data):
    return size > 1 and size or int (len (data) * size)

def split (total_xs, total_ys, test_size = 500):
    test_size = get_size (test_size, total_ys)
    train_xs, test_xs, train_ys, test_ys = train_test_split (total_xs, total_ys, test_size = test_size, random_state = random.randrange (100))
    return train_xs, test_xs, train_ys, test_ys

def split_augset (total_xs, total_ys, test_size = 500, origin_index = 0):
    test_size = get_size (test_size, total_ys)
    train_xs_0, test_xs, train_ys_0, test_ys = split (total_xs, total_ys, test_size)
    train_xs, train_ys = flatten (train_xs_0, train_ys_0)
    valid_xs, valid_ys = flatten (train_xs_0, train_ys_0, origin_index)
    test_xs, test_ys = flatten (test_xs, test_ys, origin_index)
    return train_xs, valid_xs, test_xs, train_ys, valid_ys, test_ys

def resample (batch_xs, batch_ys, sample_size = 500):
    sample_size = get_size (sample_size, batch_ys)
    sample_xs, sample_ys = [], []     
    for idx in np.random.permutation(len(batch_ys))[:sample_size]:
        sample_xs.append (batch_xs [idx])
        sample_ys.append (batch_ys [idx])        
    return np.array (sample_xs), np.array (sample_ys)

def shuffled (train_xs, count):
    norm_xs = []
    for num, idx in enumerate (np.random.permutation(len(train_xs))):
        if num > count:
            break
        norm_xs.append (train_xs [idx])
    return norm_xs    

def proportional_minibatch (xs, ys, batch_size = 64, portions = {}, infinite = False, rand = True, labels = None):
    if labels is None: 
        labels = ys
    per_class = {}    
    for idx, label in enumerate (labels):
        try: per_class [label].append (idx)
        except KeyError: per_class [label] = [idx]
    if portions and isinstance (portions, (list, tuple)):
        portions = [(idx, val) for idx, val in enumerate (portions)]
    else:    
        portions = [(label, len (idxes)) for label, idxes in per_class.items ()]
    
    recal_factor = min (1.0, batch_size / sum ([count for label, count in portions]))
    recals = [(label, round (count * recal_factor)) for label, count in portions]
    batch_size = sum ([count for label, count in recals])
    while 1:
        per_class_copy = {}
        for label, idxes in per_class.items ():
            per_class_copy [label] = idxes [:]
            if rand:
                random.shuffle (per_class_copy [label])
        while 1:
            nxs = []
            nys = []
            for label, cnt in recals:
                for i in range (cnt): 
                    try: 
                        idx = per_class_copy [label].pop ()
                    except IndexError:
                        break                
                    nxs.append (xs [idx])        
                    nys.append (ys [idx])
            if len (nys) != batch_size:                
                if not infinite:
                    raise StopIteration                    
                break
            yield np.array (nxs), np.array (nys)        

def serial_minibatch (train_xs, train_ys = None, batch_size = 0, infinite = False, rand = True):
    selectfunc = rand and np.random.permutation or np.arange
    while 1:
        if not batch_size:
            yield train_xs, train_ys        
        else:  
            pos = 0
            batch_indexes =  selectfunc (len(train_xs))
            while 1:
                batch_xs = []
                batch_ys = []
                while 1:
                    try:
                        idx = batch_indexes [pos]
                    except IndexError:
                        if not infinite:
                            raise StopIteration                    
                        pos = 0
                        batch_indexes = selectfunc (len(train_xs))
                        idx = batch_indexes [0]
                    batch_xs.append (train_xs [idx])
                    if train_ys is not None:
                        batch_ys.append (train_ys [idx])
                    pos += 1                    
                    if len (batch_xs) >= batch_size:    
                        break
                yield np.array (batch_xs), np.array (batch_ys)

def minibatch (train_xs, train_ys = None, batch_size = 0, rand = True):
    # lower version compat
    return serial_minibatch (train_xs, train_ys, batch_size, True, rand)
    
CACHE = {}
def cached_glob (path):
    global CACHE
    
    dirname = os.path.dirname (path)
    if dirname not in CACHE:
        CACHE [dirname] = {}
        for each in sorted (os.listdir (dirname)):
            if each [:4] not in CACHE [dirname]:
                CACHE [dirname][each [:4]] = []
            CACHE [dirname][each [:4]].append (each)
             
    basename = os.path.basename (path)[:-1]
    files = []
    for c4 in CACHE [dirname]:
        if basename [:4] != c4:
            continue
        for each in CACHE [dirname][c4]:
            if not each.startswith (basename):
                if files:
                    return files
                continue        
            files.append (os.path.join (dirname, each))
    return files      
    
def pad_sequences(sequences, maxlen=None, dtype='int32', padding='post', truncating='post', value=0.):
    # code from https://github.com/tflearn/tflearn/blob/master/tflearn/data_utils.py
    
    lengths = [len(s) for s in sequences]
    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    x = (np.ones((nb_samples, maxlen)) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError("Truncating type '%s' not understood" % truncating)

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError("Padding type '%s' not understood" % padding)
    return x