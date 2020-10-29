import tensorflow as tf
import numpy as np
from . import predutil
from rs4.termcolor import tc
from sklearn.utils.extmath import softmax
import math


def confidence_interval (confidence_level, error, n):
    const = {90: 1.64, 95: 1.96, 96: 2.05, 97: 2.17, 98: 2.33, 99: 2.58} [confidence_level]
    return const * math.sqrt ((error * (1 - error)) / n)

class Result:
    train_dir = None
    labels = None
    name = None
    cost_names = []
    _perfmetric_func = None
    _summary_func = None        
            
    def __init__ (self, xs, ys, logits, cost, ops, args, epoch, global_step, learning_rate, is_validating, is_evaluating, on_ftest):
        self.xs = xs
        self.ys = ys
        self.logits = logits
        self.cost = cost        
        self.epoch = epoch
        self.global_step = global_step
        self.learning_rate = learning_rate
        self.is_validating = is_validating
        self.is_evaluating = is_evaluating
        self.ops = ops
        self.args = args
        
        # set by trainer
        self.is_cherry = False
        self.on_ftest = on_ftest
        self.is_overfit = False
        self.is_improved = False
        self.performance = []
        self.performance_names = []
        self.measure_performance ()
    
    def measure_performance (self):    
        if not (self.is_evaluating and self._perfmetric_func): # no run within batch steps
            return  
          
        try:
            performance = self._perfmetric_func (self)
        except NotImplementedError:
            return        
        
        if not isinstance (performance, (tuple, list)):
            performance = [performance]
        
        for idx, p in enumerate (performance):
            if isinstance (p, (list, tuple)):
                self.performance_names.append (p [0])
                self.performance.append (p [1])
            else:
                try:
                    name = "{}-{}".format (idx + 1, self.labels [idx].name)
                except:
                    name = str (idx + 1)    
                self.performance_names.append (name)
                self.performance.append (p)        
                            
    @property
    def x (self):
        return self.xs
    
    @property
    def y (self):
        return self.ys
    
    @property
    def logit (self):
        return self.logits
    
    @property
    def phase (self):
        if not self.is_evaluating:
            return "batch"
        elif self.on_ftest:
            return "final"
        elif self.is_cherry:
            return "saved"            
        elif self.is_validating == 2:
            return "ptest"
        return self.is_validating and 'valid' or 'resub'
        
    def __make_summary (self, d = {}):
        if len (self.cost) == 1:
            d  ["eval/cost"] = self.cost [0]
        else:    
            for idx, cost in enumerate (self.cost):
                try: 
                    name = self.cost_names [idx]
                except IndexError:
                    name = str (idx + 1)
                d  ["eval/cost/{}".format (name)] = cost
        
        if len (self.performance) == 1:        
            d ["eval/acc"] = self.performance [0]
        else:      
            for idx, val in enumerate (self.performance):
                name = "eval/acc/" + self.performance_names [idx]
                d [name] = val
        
        d_ = {}
        for k, v in d.items ():
            if self.name:
                k = "{}:{}".format (self.name, k)
            if isinstance (v, (list, tuple)):
                if len (v) == 1: v = v [0]
                else: raise ValueError ("Required float, int or an array contains only one of them")
            d_ [k] = v
        return d_
    
    def __get_phase_summary (self, kargs):
        output = []
        for k, v in self.__make_summary (kargs).items ():                            
            k = k [5:] # rempve eval/
            if isinstance (v, (float, np.float64, np.float32)):
                output.append ("{} {:.5f}".format (k, v))
            elif isinstance (v, (int, np.int64, np.int32)):
                output.append ("{} {:04d}".format (k, v))
            else:
                raise ValueError ("Required float, int type")
        output.sort ()
        if self.learning_rate is not None:
            output.append ("lr {:.2e}".format (self.learning_rate))

        if self.on_ftest:
            return " | ".join (output)            
        if self.phase != "saved":
            if self.is_overfit:
                output.append (tc.error ("overfitted"))
            if self.is_improved:
                output.append (tc.info ("improved"))
        return " | ".join (output)
    
    def summary (self):
        self._summary_func (self.phase, self.__make_summary (self.args.get ("summary", {})))
        return self
    
    # statistics helper methods -----------------------------------------    
    def log (self, msg = None, **kargs):
        coloring = False
        if not msg:
            msg = self.__get_phase_summary (kargs)
            coloring = True
        phase = self.phase
        if phase == "final":
            header = "[fin.:ftest]"
            color = tc.fail
        else:    
            header = "[{:04d}:{}]".format (self.epoch, self.phase)                                  
            if phase == "saved":
                color = tc.warn
            elif phase == "ptest":
                color = tc.primary   
            else:
                color = tc.debug                   
        print ("{} {}".format ((coloring and color or tc.critical) (header), msg))
        return self

    def reunion (self, data_length, unify_func = np.mean, softmaxing = True):
        logits = []
        ys = []
        index = 0
        for part_length in data_length:        
            ys.append (self.ys [index])
            aud = self.logits [index: index + part_length]
            if softmaxing:
                aud = softmax (aud)                
            index += part_length
            logits.append (unify_func (aud, 0))
        self.logits = np.array (logits)
        self.ys = np.array (ys)
    
    def get_performance (self, index = "avg"):
        ps = self.performance
        if index == "avg":
            return np.mean (ps)
        elif index == "sum":
            return np.sum (ps)
        return ps [index] 
            
    def get_confusion_matrix (self, logits = None, ys = None):
        if logits is None:
            logits = self.logits
        if ys is None:
            ys = self.ys
        mat_ = predutil.confusion_matrix (
            np.argmax (logits, 1), 
            np.argmax (ys, 1),
            logits.shape [1]
        )
        return mat_.T
    
    def slice_by_label (self, label_index):
        start_index = 0
        for i in range (label_index + 1):
            num_label = len (self.labels [i])
            if label_index == i:
                break
            start_index += num_label        
        logits = self.logits [:, start_index:start_index + num_label]
        if len (self.y.shape) == 1: # sparse categorical encoding           
            ys = np.zeros ([len (self.ys), len (self.labels [label_index])])
            for row, y in enumerate (self.ys):
                ys [row][y] = 1.0
        else:        
            ys = self.ys [:, start_index:start_index + num_label]
            if len (ys [0]) == 0:
                ys = self.ys [:, -num_label:]
        return logits, ys
    
    @property    
    def metric (self):
        class Metric:
            def __init__ (self):
                class Tri:
                    def __init__ (self):
                        self.micro = []
                        self.macro = []
                        self.weighted = []                        
                self.accuracy = []
                self.precision = Tri ()
                self.recall = Tri ()
                self.f1 = Tri ()
            
        metric = Metric ()
        if self.labels is None:
            self.calculate_metric (metric, self.logits, self.ys)
        else:            
            for idx, label in enumerate (self.labels):
                logits, ys = self.slice_by_label (idx)
                self.calculate_metric (metric, logits, ys)
        return metric
    
    def calculate_metric (self, metric, logits, ys):
        mat = self.get_confusion_matrix (logits, ys)
        catpreds = mat.sum (axis = 0)
        table = np.zeros ([mat.shape [0], 3])
        for idx, preds in enumerate (mat.T):
            tp = preds [idx]
            fp = catpreds [idx] - tp            
            table [idx][0] = tp
            table [idx][1] = fp
            
        catans = mat.sum (axis = 1)
        for idx, ans in enumerate (mat):
            tp = ans [idx]
            fn = catans [idx] - tp
            table [idx][2] = fn                
        
        accuracy = np.sum (table [:, 0]) / len (logits)
        micro_precision = np.sum (table [:, :1]) / np.sum (table [:, :2])
        micro_recall = np.sum (table [:, :1]) / np.sum ([table [:, 0], table [:, 2]])
        metric.accuracy.append (accuracy)                        
        metric.precision.micro.append (micro_precision)
        metric.recall.micro.append (micro_recall)
        metric.f1.micro.append (2 / (1 / micro_precision + 1 / micro_recall))
        
        precisions = []
        recalls = []            
        f1s = []
        alpha = 1e-8
        for cat, (tp, fp, fn) in enumerate (table):
            precision = tp / (tp + fp + alpha)                
            recall = tp / (tp + fn + alpha)
            f1 = (2 * (precision * recall)) / (precision + recall + alpha)
            precisions.append (precision)
            recalls.append (precision)
            f1s.append (f1)
        
        metric.precision.macro.append (np.mean (precisions))
        metric.precision.weighted.append (np.average (precisions, weights = catans))
        metric.recall.macro.append (np.mean (recalls))                    
        metric.recall.weighted.append (np.average (recalls, weights = catans))
        metric.f1.macro.append (np.mean (f1s))
        metric.f1.weighted.append (np.average (f1s, weights = catans))
            
    def confusion_matrix (self, num_label = 0, label_index = 0, indent = 13, show_label = True):
        start_index = 0
        name = None
        if num_label == 0:
            if not self.labels:
                raise ValueError ("num_label required")            
            logits, ys = self.slice_by_label (label_index)
            name = self.labels [label_index].name
        else:
            logits = self.logits [:, :num_label]
            ys = self.ys [:, :num_label]
            
        mat_ = self.get_confusion_matrix (logits, ys)        
        mat = str (mat_) [1:-1]
        self.log ("confusion matrix{}".format (name and (" of " + name) or ""))
        
        labels = []
        if show_label and self.labels:
            cur_label = self.labels [label_index]
            first_row_length = len (mat.split ("\n", 1) [0]) - 2
            label_width = (first_row_length - 1) // mat_.shape [-1]
            labels = [str (each) [:label_width].rjust (label_width) for each in cur_label.items ()]            
            print (tc.fail ((" " * (indent + label_width + 1)) + " ".join (labels)))
            
        lines = []
        for idx, line in enumerate (mat.split ("\n")):
            if idx > 0:
                line = line [1:]
            line = line [1:-1]                    
            if labels:    
                line = tc.info (labels [idx]) + " " + line
            if indent:    
                line = (" " * indent) + line
            print (line)

    def logit_range (self):
        output_range = [self.logit.min (), self.logit.max (), np.mean (self.logit), np.std (self.logit)]
        quant_range = {}
        for idx, m in enumerate (self.logit [:,:29].argsort (1)[:,-2]):
            sec = int (self.logit [idx, m])
            try: quant_range [sec] += 1
            except KeyError: quant_range [sec] = 1
        quant_range = quant_range
        if quant_range:           
            stats = sorted (quant_range.items ())
            # output range for top1: {} ~ {}, logit range: {:.3f} ~ {:.3f}, mean: {:.3f} std: {:.3f}
            return stats [0][0] - 1, stats [-1][0] + 1, output_range [0], output_range [1], output_range [2], output_range [3]                                
        
    def confidence_level (self, label_index = 0):
        label = self.labels [label_index]
        stat = {}
        for idx in range (len (self.y)):
            logit = self.logit [idx][:len (label)]
            y = self.y [idx][:len (label)]
            probs = predutil.softmax (logit)
            prob = "{:.5f}".format (probs [np.argmax (probs)])
            if prob not in stat:
                stat [prob] = [0, 0]
            if np.argmax (probs) == np.argmax (y):    
                stat [prob][0] += 1
            stat [prob][1] += 1
            
        ordered = [] 
        accum = 0        
        accum_t = 0
        total = len (self.y)
        for lev, (c, t) in sorted (stat.items (), reverse = True):            
            accum += c            
            accum_t += t
            accuracy = accum / accum_t * 100
            ordered.append ((
                lev, 
                accum, accum_t, accuracy, 
                accum / total * 100 
            ))
            if len (ordered) >= 10 and accuracy < 100.:
                break
        return ordered    
    
