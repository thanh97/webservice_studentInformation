import numpy as np
import math
from sklearn.utils.extmath import softmax as softmax_
import hyperopt
import os
from rs4 import siesta

def softmax (x):
    x = np.array (x)
    if len (x.shape) == 2:
        return softmax_ (x).tolist ()
    return softmax_ ([x])[0].tolist ()

def sigmoid (x):
    return [1 / (1 + np.exp(-e)) for e in x]

def confusion_matrix (labels, predictions, num_labels):
  rows = []
  for i in range (num_labels):
    row = np.bincount (predictions[labels == i], minlength=num_labels)
    rows.append (row)
  return np.vstack (rows)

def render_trial (space, trial, stringfy = False):
    if trial.get ("misc"):
        trial = dict ([(k, v [0]) for k, v in trial ["misc"]["vals"].items ()])
    params = hyperopt.space_eval(space, trial)
    if stringfy:
        return ", ".join (["{}: {}".format (k, v) for k, v in params.items ()])
    return params

def get_latest_model (path):
    if not os.path.isdir (path) or not os.listdir (path):
        return
    version = max ([int (ver) for ver in os.listdir (path) if ver.isdigit () and os.path.isdir (os.path.join (path, ver))])    
    return os.path.join (path, str (version))
 
 
# Model Loaders ------------------------------------------------

class CheckPoint:
    def __init__ (self, net):
        self.net = net
    
    def ftest (self, xs, ys, **kargs):
        return self.net.ftest (self.net.normalize (xs), ys, **kargs)
            
    def predict (self, xs):
        return self.net.run (self.net.logit, x = self.net.normalize (xs)) [0]
        

class SavedModel:
    def __init__ (self, model_root, debug = False):
        self.model_root = model_root
        self.debug = debug
        self.model_path = get_latest_model (self.model_root)        
        self.stub = self.create_stub (self.model_path)
        self.version = os.path.basename (self.model_path)        
    
    def create_stub (self, model_path):
        from . import saved_model
        return saved_model.load (model_path)
                
    def predict (self, x, **kargs):
        y = self.stub.run (x, **kargs)[0]
        return y.astype (np.float64)

    
class TFLite (SavedModel):    
    def __init__ (self, model_root, model_file = "model.tflite", debug = False):
        self.model_file = model_file
        SavedModel.__init__ (self, model_root, debug)
    
    def create_stub (self, model_path):
        from . import tflite
        
        return tflite.load (
            os.path.join (model_path, self.model_file),
            (128, 128)
        )
    
    def predict (self, x, **kargs):   
        ys = []
        for x_ in x:     
            ys.append (self.stub.run (x_, **kargs)[0].astype (np.float64))
        return np.array (ys)    


class TFServer:
    def __init__ (self, endpoint, alias, debug = False):
        from tfserver import cli
                
        self.endpoint = endpoint
        self.alias = alias
        self.debug = debug
        
        self.stub = cli.Server (endpoint)
        api = siesta.API (endpoint)
        resp = api.model (self.alias).version.get ()
        self.version = resp.data.version         
        
    def predict (self, x, **kargs):
        resp = self.stub.predict (self.alias, 'predict', x = x, **kargs)
        return resp.y
            
