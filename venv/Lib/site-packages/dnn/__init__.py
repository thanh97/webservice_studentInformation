from .dnn import DNN
from .multidnn import MultiDNN  

__version__ = "0.3.5.7"

def subprocess (train_func):
    from multiprocessing import Pool    
    with Pool(1) as p:
        return p.apply (train_func, ())
    
