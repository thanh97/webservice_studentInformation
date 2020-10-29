# required tensorflow >= 1.9 

from tensorflow.contrib import lite
from tensorflow.python.tools import freeze_graph
import tensorflow as tf
import numpy as np
import threading
from tensorflow.contrib.lite.python.convert import tensor_name
from . import saved_model
import os

"""
freeze_graph --input_checkpoint=resources/chkpoint/0025-acc-87.267+cost-2.988-43750 \
    --input_graph=resources/exported/toco/graph-def.pb \
    --input_binary=true \
    --output_node_names=concat_1:0 \
    --output_graph=resources/exported/toco/frozen-graph.pb

toco --graph_def_file=resources/exported/lge7/70/frozen-graph.pb \
  --output_format=TFLITE \
  --output_file=resources/exported/lge7/70/model.tflite \
  --input_arrays=Placeholder \
  --output_arrays=concat_1 \
  --inference_type=QUANTIZED_UINT8 \
  --inference_input_type=QUANTIZED_UINT8 \
  --std_dev_values=128 \
  --mean_values=128 \
  --default_ranges_min=-1 \
  --default_ranges_max=6 \
  --allow_nudging_weights_to_use_fast_gemm_kernel

toco --graph_def_file=resources/exported/lge7/70/frozen-graph.pb \
  --output_format=TFLITE \
  --output_file=resources/exported/lge7/70/model.tflite \
  --input_arrays=Placeholder \
  --output_arrays=concat_1
"""

def convert (path, saved_model_dir, quantize = False, quantized_input_stats = (128, 128), default_ranges_stats = (0, 1)):
    try:
        converter = tf.contrib.lite.TFLiteConverter.from_saved_model (saved_model_dir)
    except AttributeError:
        converter = tf.contrib.lite.TocoConverter.from_saved_model (saved_model_dir)
    if quantize:
        converter.inference_type = tf.contrib.lite.constants.QUANTIZED_UINT8
        converter.quantized_input_stats = {tensor_name (converter._input_tensors [0]): quantized_input_stats}
        converter.default_ranges_stats =  default_ranges_stats
        converter.allow_nudging_weights_to_use_fast_gemm_kernel = True
                    
    tflite_model = converter.convert()
    open(path, "wb").write(tflite_model)


def convert_from_graph_def (train_dir, graph_def_file, input_arrays, output_arrays, quantize = False, quantized_input_stats = (128, 128), default_ranges_stats = (0, 1)):
    checkpoints = [] 
    for f in os.listdir (train_dir):
        s = f.find (".data-")
        if s == -1:
            continue
        checkpoints.append (f [:s])
    
    checkpoints.sort (key = lambda x: int (x.split ("-")[-1]))
    output_graph = os.path.join (os.path.dirname (graph_def_file), "frozen-graph-def.pb")
    output_model = os.path.join (os.path.dirname (graph_def_file), "model.tflite")
    
    cmd = [
        "freeze_graph --input_checkpoint={}".format (os.path.join (train_dir, checkpoints [-1])),
        "--input_graph={} --input_binary=true".format (graph_def_file),
        "--output_node_names={}".format (",".join (output_arrays)),
        "--output_graph={}".format (output_graph)        
    ]
    os.system (" ".join (cmd))
    
    cmd = [    
      "tflite_convert --graph_def_file={}".format (output_graph),
      "--output_format=TFLITE",
      "--output_file={}".format (output_model),      
      "--input_arrays={}".format (".".join (input_arrays)),
      "--output_arrays={}".format (".".join (output_arrays))
    ]
    if quantize:
        cmd.append ("--inference_type=QUANTIZED_UINT8")
        cmd.append ("--inference_input_type=QUANTIZED_UINT8")
        cmd.append ("--mean_values={}".format (quantized_input_stats [0]))
        cmd.append ("--std_dev_values={}".format (quantized_input_stats [1]))
        cmd.append ("--default_ranges_min={}".format (default_ranges_stats [0]))
        cmd.append ("--default_ranges_max={}".format (default_ranges_stats [1]))
        cmd.append ("--allow_nudging_weights_to_use_fast_gemm_kernel=true")   

    os.system (" ".join (cmd))
    
def load (path, quantized = None):
    return Interpreter (path, quantized)


class Interpreter (saved_model.Interpreter):    
    def __init__ (self, path, quantized = None, debug = False):        
        self.path = path
        self._quantized = quantized
        self.debug = debug
        self.interp = lite.Interpreter (path)
        self.model_dir = os.path.dirname (path)
        self.norm_factor = self.load_norm_factor ()
        self.input = self.interp.get_input_details()[0]["index"]
        self.output = self.interp.get_output_details()[0]["index"]
        self.lock = threading.RLock ()
        if self.is_quantized ():
            assert self._quantized and len (self._quantized) == 2, "quantized should be 2 integer tuple like (128, 128)"
        else:
            self._quantized = None
        self._allocated = False
        
    def is_quantized (self):
        return self.interp.get_input_details()[0]["dtype"] is np.uint8
        
    def get_info (self): 
        inputs = {}
        outputs = {}
        for each in self.interp.get_input_details ():
            inputs [each ["name"]] = (each ["index"], each ["shape"], each ["dtype"])
        for each in self.interp.get_output_details ():
            outputs [each ["name"]] = (each ["index"], each ["shape"], each ["dtype"])     
        return inputs, outputs      
        
    def run (self, x, **kargs):
        if not self._allocated:
            if self._quantized:
                self.interp.resize_tensor_input (self.input, np.array(x.shape, dtype=np.int32))
            self.interp.allocate_tensors ()
            self._allocated = True
        
        if self.debug:
            print ("unorm", x [0][0])
        x =  self.normalize (x)
        if self.debug:
            print ("normed", x [0][0])
        if self._quantized:
            x = x * self._quantized [0] + self._quantized [1]
            x = np.clip (x, 0, 255).astype ("uint8")            
            if self.debug:
                print ("qunatized", x [0][0])
        else:
            x = x.astype ("float32")
        
        with self.lock:
            self.interp.set_tensor (self.input, x)
            self.interp.invoke ()
            y = self.interp.get_tensor (self.output)
        return y
