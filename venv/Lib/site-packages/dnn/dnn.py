import tensorflow as tf
import numpy as np
import sys
import os, shutil
import random
from rs4 import pathtool
from . import overfit, optimizers, predutil, result, normalizer, label
import sys
from functools import partial
import pickle
from sklearn.decomposition import PCA
from tensorflow.python.framework.ops import Tensor
import warnings
from rs4.termcolor import tc
import copy

class DNN:
    def __init__ (self, gpu_usage = 0, name = None, graph = None):
        self.gpu = gpu_usage
        self.name = name
        result.Result.name = name

        if graph is None:
            self.graph = tf.Graph ()
        else:
            self.graph = graph
        self.sess = None
        self.in_service = True

        self.norm_factor = None
        self.norm_file = None
        self.labels = None
        self.label_file = None

        self.trainabled = False
        self.initialize_training_related_variables ()

    def initialize_training_related_variables (self):
        if self.trainabled:
            return

        self.cost = None
        self.verbose = True
        self.filter_func = None

        self.writers = {}
        self.summaries_dir = None
        self.log_per_steps = 100

        self.auto_save = True
        self.train_dir = None
        self.max_performance = 0.0
        self.performance_threshold = 0.0
        self.plateau_continued = 0
        self.overfitwatch = None
        self.metric = "cost"
        self.metric_index = 0

        self.initial_learning_rate = None
        self.__optimzables = set ()
        self.is_validating = False
        self.is_improved = False
        self.on_ftest = False
        self.batch_size = 32
        self.epoch = 0
        self.cherry = None

        self.is_major_optimizer = True
        self.trainabled = True

    def create_network (self):
        with self.graph.as_default ():
            self.before_graph_create ()
            with tf.variable_scope ("placeholders"):
                self.make_default_place_holders ()
                self.make_place_holders ()
            self.make_variables ()
            self.logit = self.make_logit ()
            if isinstance (self.logit, tuple):
                self.logit, self.end_points = self.logit
            self.graph_created ()
            self.saver = tf.train.Saver (tf.global_variables())

    def make_default_place_holders (self):
        self.seq_length = None
        self.dropout_rate = tf.placeholder_with_default (tf.constant (0.0), [])
        self.is_training = tf.placeholder_with_default (tf.constant (False), [])
        self.n_sample = tf.placeholder_with_default (tf.constant (1), [])
        self.random_dropout_rate = tf.random_uniform ([], minval=0.1, maxval=0.7, dtype=tf.float32)
        self.nullop = tf.constant (0.0)

    def use_seq_length (self):
        self.seq_length = tf.placeholder (tf.int32, [None])

    def init_session (self):
        if self.sess is not None:
            return
        if self.gpu:
            self.sess = tf.Session (graph = self.graph, config = tf.ConfigProto(gpu_options=tf.GPUOptions (per_process_gpu_memory_fraction = self.gpu), log_device_placement = False))
        else:
            self.sess = tf.Session(graph = self.graph)
        self.sess.run (tf.global_variables_initializer())

    def get_best_cost (self):
        return overfitwatch.min_cost

    def  get_best_performance (self):
        return self.name, self.max_performance

    @property
    def is_overfit (self):
        return self.overfitwatch.is_overfit ()

    def eval (self, tensor):
        with self.sess.as_default ():
            return tensor.eval ()

    # data filtering for multi model training -----------------------------
    def set_filter (self, func):
        self.filter_func = func

    def filter (self, ys, *args):
        is_no_x = True
        xs = None
        if args:
            is_no_x = False
            xs = args [0]

        if self.filter_func:
            ys, xs = self.filter_func (ys, xs)
        if is_no_x:
            return ys
        return ys, xs

    # labels -----------------------------------------------------------
    def set_labels (self, labels):
        if not isinstance (labels, (list, tuple, label.Labels)):
            labels = [labels]
        self.labels = labels
        result.Result.labels = self.labels

    def save_labels (self):
        with open (self.label_file, "wb") as f:
            pickle.dump (self.labels, f)

    def load_labels (self):
        if not os.path.isfile (self.label_file):
            return
        with open (os.path.join (self.label_file), "rb") as f:
            self.labels = pickle.load (f)

    def update_epoch (self):
        self.epoch += 1

    def set_epoch (self, epoch = 0):
        if epoch == 0:
            self.update_epoch ()
        else:
            self.epoch = epoch

    def get_epoch (self):
        return self.epoch

    # normalization -----------------------------------------------------
    def get_norm_factor (self):
        return self.norm_factor

    def has_norm_factor (self):
        return os.path.exists (self.norm_file)

    def load_norm_factor (self):
        if self.norm_factor:
            return
        if not os.path.isfile (self.norm_file):
            return
        with open (self.norm_file, "rb") as f:
            self.norm_factor = pickle.load (f)

    ITERSIZE = 10000
    def normalize (self, x, normalize = False, standardize = False, axis = 0, pca_k = None, pca_random = False):
        if not len (x):
            return x
        if self.norm_factor:
            len_origin = len (x)
            stack = []
            for i in range (0, len (x), self.ITERSIZE):
                q, x = x [:i + self.ITERSIZE], x [i + self.ITERSIZE:]
                stack.append (normalizer.normalize (q, *self.norm_factor))
            if len (stack) == 1:
                return stack [0]
            x = np.vstack (stack)
            assert (len_origin == len (x))
            return x

        if not normalize and not standardize and pca_k is None:
            if not self.norm_file:
                return x
            if os.path.isfile (self.norm_file):
                os.remove (self.norm_file)
            return x

        min0_ = None
        gap0 = None

        if isinstance (x, list):
            x = np.array (x)

        mean = np.mean (x, axis, keepdims = True)
        std = np.std (x, axis, keepdims = True) + 1e-8
        if standardize: # 0 mean, 1 var
            x = normalizer.standardize (x, mean, std)

        min_ = np.min (x, axis, keepdims = True)
        gap = (np.max (x, axis, keepdims = True) - min_) + 1e-8
        if normalize: # -1 to 1
            x = normalizer.scaling (x, min_, gap, normalize)

        eigen_vecs = None
        pca_mean = None
        if pca_k:
            if pca_k < 0:
                self.show_pca (x, pca_random)
            else:
                x, pca = self.pca (x, pca_k, pca_random)
            eigen_vecs = pca.components_.swapaxes (1, 0)
            pca_mean = pca.mean_
            # DO NOT NORMALIZE pca transformed data

        self.norm_factor = (mean, std, min_, gap, pca_k, eigen_vecs, pca_mean, normalize, standardize)
        if self.norm_file and (normalize or standardize or pca_k):
            with open (self.norm_file, "wb") as f:
                pickle.dump (self.norm_factor, f)

        return x

    def show_pca (self, data, pca_random = False):
        orig_shape = data.shape
        if len (orig_shape) == 3:
            data = data.reshape ([orig_shape [0]  * orig_shape [1], orig_shape [2]])

        print ("* Principal component analyzing (showing eigen vector)...")
        pca = PCA (n_components = orig_shape [-1], svd_solver = pca_random and 'randomized' or "auto")
        pca.fit (data)
        for i, r in enumerate (pca.explained_variance_ratio_.cumsum ()):
            if r > 0.9 and i % 10 == 0:
                print ("n_components: {}, retained variance: {:.2f}".format (i, r))
                if "{:.2f}".format (r) == "1.00":
                    break
        print ("* Principal component analyzing, done.")
        sys.exit (1)

    def pca (self, data, n_components = None, pca_random = False):
        orig_shape = data.shape
        if len (orig_shape) == 3:
            data = data.reshape ([orig_shape [0]  * orig_shape [1], orig_shape [2]])
        pca = PCA (n_components = n_components, svd_solver = pca_random and 'randomized' or "auto")
        pca.fit (data)
        data = pca.transform (data)
        if len (orig_shape) == 3:
            data = data.reshape ([orig_shape [0], orig_shape [1], n_components])
        return data, pca

    # train dir / log dir ----------------------------------------------------
    def turn_off_verbose (self):
        self.verbose = False

    def reset_dir (self, target):
        if os.path.exists (target):
            shutil.rmtree (target)
        if not os.path.exists (target):
            os.makedirs (target)

    def set_train_dir (self, path, reset = False, auto_save = True, improve_metric = "cost", performance_threshold = 0.0):
        self.auto_save = auto_save
        self.performance_threshold = performance_threshold

        if not (improve_metric.startswith ("cost") or improve_metric.startswith ("performance")):
            raise ValueError ("improve_metric should be one of cost:index|performance[:index]")

        try:
            self.metric, self.metric_index = improve_metric.split (":", 1)
        except (ValueError):
            self.metric, self.metric_index = improve_metric, "avg"
        try:
            self.metric_index = int (self.metric_index)
        except ValueError:
            pass

        if self.name:
            path = os.path.join (path, self.name.strip ())

        self.train_dir = path
        if reset and os.path.exists (self.train_dir):
            for file in os.listdir (self.train_dir):
                if file == "normfactors":
                    continue
                t = os.path.join (self.train_dir, file)
                if os.path.isdir (t):
                    shutil.rmtree (t)
                else:
                    os.remove (t)
        else:
            pathtool.mkdir (self.train_dir)

        self.norm_file = os.path.join (self.train_dir, 'normfactors')
        self.label_file = os.path.join (self.train_dir, 'labels')
        result.Result.train_dir = self.train_dir

    def set_tensorboard_dir (self, summaries_dir, reset = True, log_per_steps = 10):
        self.summaries_dir = summaries_dir
        self.log_per_steps = log_per_steps
        if reset:
            os.system ('killall tensorboard')
            if tf.gfile.Exists(summaries_dir):
                tf.gfile.DeleteRecursively(summaries_dir)
            tf.gfile.MakeDirs(summaries_dir)

    def get_writers (self, *writedirs):
        return [tf.summary.FileWriter(os.path.join (self.summaries_dir, "%s%s" % (self.name and self.name.strip () + "-" or "", wd)), self.graph) for wd in writedirs]

    def make_writers (self, *writedirs):
        for i, w in enumerate (self.get_writers (*writedirs)):
            self.writers [writedirs [i]] = w

    def write_summary (self, writer, feed_dict):
        if self.on_ftest or int (self.is_validating) > 1:
            return
        summary = tf.Summary()
        for k, v in feed_dict.items ():
            summary.value.add (tag = k , simple_value = v)
        if not self.on_ftest and writer not in self.writers:
            self.make_writers (writer)
        self.writers [writer].add_summary (summary, self.eval (self.global_step))

    def log (self, name, val, family = "train"):
        if family == "ptest":
            return
        if not self.summaries_dir:
            return
        if isinstance (val, Tensor):
            #tf.summary.scalar (name, val)
            tf.summary.scalar ("train/" + name, self.add_average (val))
        elif self.sess:
            self.write_summary (family, {name: val}, False)

    def logp (self, name, val):
        tag = "resub"
        if self.is_validating == 2:
            tag = "ptest"
        elif self.is_validating:
            tag = "valid"
        self.log (name, val, tag)

    def add_average(self, variable):
        tf.add_to_collection (tf.GraphKeys.UPDATE_OPS, self.ema.apply([variable]))
        average_variable = tf.identity (self.ema.average(variable), name=variable.name[:-2] + '_avg')
        return average_variable

    # model save -------------------------------------------------------
    def restore (self, for_train  = False):
        import glob

        self.load_norm_factor ()
        self.load_labels ()
        self.create_network ()
        if for_train:
            with open (os.path.join (self.train_dir, "progress"), "rb") as f:
                global_step, epoch = pickle.load (f)
            with open (os.path.join (self.train_dir, "objects"), "rb") as f:
                 self.max_performance, self.overfitwatch, self.cherry = pickle.load (f)
            self.create_trainables (global_step, epoch)

        with self.graph.as_default ():
            self.init_session()
        with self.graph.as_default ():
            self.saver.restore (self.sess, tf.train.latest_checkpoint (self.train_dir))

    def save (self, filename = None):
        if not filename:
            if self.overfitwatch.validations:
                filename = "%04d-acc-%.3f+cost-%.3f" % (self.epoch, self.max_performance, self.overfitwatch.latest_cost)
            else:
                filename = "%04d" % (self.epoch)
        path = os.path.join (self.train_dir, filename)
        with self.graph.as_default ():
            self.saver.save (self.sess, path, global_step = self.global_step)
        with open (os.path.join (self.train_dir, "objects"), "wb") as f:
            pickle.dump ((self.max_performance, self.overfitwatch, self.cherry), f)

    def get_latest_model_version (self, path):
        if not os.listdir (path):
            return 0
        return max ([int (ver) for ver in os.listdir (path) if ver.isdigit () and os.path.isdir (os.path.join (path, ver))])

    def to_tflite (self, path, saved_model_dir, quantized_input = None, quantized_input_stats = (128, 128), default_ranges_stats = (0, 1)):
        from . import tflite

        tflite.convert (path, saved_model_dir, quantized_input, quantized_input_stats, default_ranges_stats)
        interp = tflite.Interpreter (path, quantized_input is not None and quantized_input_stats or None)
        inputs, outputs = interp.get_info ()

        print ("* TF Lite")
        #print ("  - " + self.output_stat (os.path.join (saved_model_dir, "outputstat")))
        print ("  - Inputs")
        for k, v in inputs.items (): print ("    . {}: {}".format (k, v))
        print ("  - Outputs")
        for k, v in outputs.items (): print ("    . {}: {}".format (k, v))

    def to_tflite_from_graph_def (self, saved_model_dir, inputs, outputs, quantized_input = None, quantized_input_stats = (128, 128), default_ranges_stats = (0, 1)):
        from . import tflite

        inputs=dict ([(k, tf.saved_model.utils.build_tensor_info (v)) for k, v in inputs.items ()])
        outputs=dict ([(k, tf.saved_model.utils.build_tensor_info (v)) for k,v in outputs.items ()])

        graph_def_file = os.path.join (saved_model_dir, 'graph-def.pb')
        if not os.path.isfile (graph_def_file):
            tf.train.write_graph (self.sess.graph_def, saved_model_dir, 'graph-def.pb', as_text=False)
        input_arrays = [v.name.endswith (":0") and v.name [:-2] or v for v in inputs.values ()]
        output_arrays = [v.name.endswith (":0") and v.name [:-2] or v for v in outputs.values ()]
        tflite.convert_from_graph_def (self.train_dir, graph_def_file, input_arrays, output_arrays, quantized_input, quantized_input_stats, default_ranges_stats)

    def to_constatn_model (self):
        # deprecated on TF 2.0
        from tensorflow.python.framework import graph_util
        from tensorflow.python.framework import graph_io

        if self.name:
            path = os.path.join (path, self.name.strip ())
        pathtool.mkdir (path)
        version = self.get_latest_model_version (path) + 1

        cgraph = graph_util.convert_variables_to_constants (self.sess, self.graph.as_graph_def(), [t.name.split (":")[0] for t in outputs.values ()])
        graph_io.write_graph(cgraph,  "{}/{}/".format (path, version), 'constant_model.pb', as_text = False)

    def to_save_model (self, path, predict_def, inputs, outputs):
        from . import saved_model

        if self.name:
            path = os.path.join (path, self.name.strip ())
        pathtool.mkdir (path)
        version = self.get_latest_model_version (path) + 1

        dirname = "{}/{}/".format (path, version)
        inputs, outputs = saved_model.convert (
            dirname,
            predict_def, inputs, outputs,
            self.sess,
            self.graph
        )
        tf.train.write_graph (self.graph.as_graph_def(), dirname, 'graph-def.txt', as_text = True)
        tf.train.write_graph (self.graph.as_graph_def(), dirname, 'graph-def.pb', as_text = False)
        if os.path.isfile (self.norm_file):
            shutil.copy (self.norm_file, os.path.join (path, str (version), "normfactors"))
        if os.path.isfile (self.label_file):
            shutil.copy (self.label_file, os.path.join (path, str (version), "labels"))

        print ("* Saved Model")
        print ("  - Inputs")
        for k, v in inputs.items (): print ("    . {}: {}".format (k, v.name))
        print ("  - Outputs")
        for k, v in outputs.items (): print ("    . {}: {}".format (k, v.name))
        return version

    export = to_save_model

    def maybe_save_checkpoint (self, r):
        performance = r.get_performance (self.metric == "cost" and "avg" or self.metric_index)
        if self.metric == 'performance' and performance < self.performance_threshold:
            return
        save = False
        if self.metric == 'cost' and self.overfitwatch.is_renewaled ():
            save = True
        if performance > self.max_performance:
            self.max_performance = performance
            if self.metric == "performance":
                save = True

        if save:
            self.is_improved = True
            if self.auto_save and self.train_dir:
                self.save ()
                with open (os.path.join (self.train_dir, "progress"), "wb") as f:
                    pickle.dump ((self.eval (self.global_step), self.epoch), f)
            self.plateau_continued = 0

        elif self.decay_patience:
            self.plateau_continued += 1
            if self.plateau_continued >= self.decay_patience:
                self.sess.run (self.__increase_lr_reduce_step_op)
                self.plateau_continued = self.decay_cooldown
                if self.eval (self.learning_rate) <= self.min_lr:
                    self.decay_patience = 0

    # make trainable ----------------------------------------------------------
    def get_regularization_losses (self, scopes = [None]):
        if not isinstance (scopes, (tuple, list)):
            scopes = [scopes]
        losses = 0.0
        for scope in scopes:
            losses += tf.losses.get_regularization_loss (scope)
        return losses

    def set_learning_rate (self, initial_learning_rate = 0.01, decay_rate = 1.0, decay_step = 0, decay_patience = 0, decay_cooldown = 0, min_lr = 1e-6):
        if decay_step and decay_patience:
            raise ValueError ("decay_step and decay_patience should be mutaully exclusive")
        self.initial_learning_rate = initial_learning_rate
        self.decay_rate = decay_rate
        self.decay_step = decay_step
        self.decay_patience = decay_patience
        self.decay_cooldown = decay_cooldown
        self.min_lr = min_lr

    def set_cherry (self, r):
        self.cherry = copy.copy (r)
        self.cherry.is_cherry = True

    def early_terminate (self, patience, min_validations = 0, period = 5):
        if self.overfitwatch is None:
            self.overfitwatch = overfit.Overfit (patience, period, min_validations)

    def add_cost (self):
        costs = self.make_cost ()
        if hasattr (costs, "dtype"):
            costs = [costs]
        elif isinstance (costs, (list, tuple)):
            if not isinstance (costs [0], (list, tuple)):
                costs = [costs]

        self.cost = []
        for cost in costs:
            if isinstance (cost, tuple):
                cost, reg_cost, var_list, task_name = cost
            else:
                reg_cost = tf.add (cost, self.get_regularization_losses ())
                var_list = []
                task_name = ""

            self.cost.append ((cost, reg_cost, var_list, task_name))
            self.log ("cost{}".format (task_name and ("/" + task_name) or ""), cost)
            self.log ("cost{}/reg".format (task_name and ("/" + task_name) or ""), reg_cost)
        result.Result.cost_names = [cost [-1] for cost in self.cost]

    def add_performance (self):
        result.Result._summary_func = self.write_summary
        result.Result._perfmetric_func = self.performance_metric

    def create_trainables (self, global_step = 0, epoch = 0):
        self.epoch = epoch
        if not self.summaries_dir:
            self.set_tensorboard_dir ("/var/tmp/tflog", True)
        self.make_writers ("train")
        self.in_service = False

        with self.graph.as_default ():
            self.global_step = tf.Variable (global_step, trainable = False)
            self.ema = tf.train.ExponentialMovingAverage(0.99, self.global_step)
            if self.decay_step:
                self.learning_rate = tf.train.exponential_decay (
                    self.initial_learning_rate,
                    self.global_step,
                    self.decay_step,
                    self.decay_rate,
                    staircase = True
                )
            else:
                self.lr_reduce_step = tf.Variable (0.0, trainable = False)
                self.__increase_lr_reduce_step_op = self.lr_reduce_step.assign_add (1.0)
                self.learning_rate = self.initial_learning_rate * tf.pow (1.0 - (1.0 - self.decay_rate), self.lr_reduce_step)

            self.log ("learning-rate", self.learning_rate)
            self.add_cost ()

            self.update_ops = tf.get_collection (tf.GraphKeys.UPDATE_OPS)
            self.optimize_op = self.make_optimizer ()
            if self.__optimzables:
                for var in tf.trainable_variables ():
                    if var not in self.__optimzables:
                        raise AssertionError ("{} will be not optimized".format (var.name))

            self.add_performance ()
            self.summary_op = tf.summary.merge_all ()

    def setup (self):
        if self.sess is not None:
            return
        if self.initial_learning_rate is None:
            raise RuntimeError ("call set_learning_rate () first")
        if self.overfitwatch is None:
            self.overfitwatch = overfit.Overfit (0)
        if self.labels and self.label_file:
            self.save_labels ()

        self.create_network ()
        self.create_trainables ()
        with self.graph.as_default ():
            self.init_session()
        self.train_setuped ()

    def count_kernels (self, scopes = []):
        if not isinstance (scopes, (tuple, list)):
            scopes = [scopes]

        kernels = []
        with self.graph.as_default ():
            for variable in tf.trainable_variables ():
                valid = True
                if scopes:
                    valid = False
                    for scope in scopes:
                        if variable.name.startswith (scope):
                            valid = True
                            break
                #if variable.name.find ("/kernel") == -1:
                #    valid = False
                if not valid:
                    continue
                if variable.name.find ("kernel") == -1:
                    continue
                shape = variable.get_shape()
                kernels.append ((variable.name, shape))
        return kernels

    def get_global_step (self):
        global_step = None
        if hasattr (self, "global_step"):
            global_step = self.eval (self.global_step)
        return global_step

    def get_learning_rate (self):
        learning_rate = None
        if hasattr (self, "learning_rate"):
            learning_rate = self.eval (self.learning_rate)
        return learning_rate

    def evaluate (self, x, y, is_training, ops, *args, **kargs):
        logits = []
        costs = []
        opsres = []
        ops = [self.logit, [cost [1] for cost in self.cost]] + ops

        seq_length = None
        if "seq_length" in kargs:
            seq_length = kargs.pop ("seq_length")

        for i in range (0, len (x), self.batch_size):
            x_ = x [i:i + self.batch_size]
            y_ = y [i:i + self.batch_size]
            if seq_length is not None:
                seq_length_ = seq_length [i:i + self.batch_size]
                results =  self.run (*ops, x = x_, y = y_, dropout_rate = 0.0, is_training = is_training, seq_length = seq_length_, **kargs)
            else:
                results =  self.run (*ops, x = x_, y = y_, dropout_rate = 0.0, is_training = is_training, **kargs)
            logits.append (results [0])
            costs.append (results [1])
            opsres.append (results [2:])

        r = result.Result (
            x, y, np.concatenate(logits, 0),
            np.mean (costs, 0),
            np.concatenate(opsres, 0), kargs,
            self.epoch,
            self.get_global_step (), self.get_learning_rate (),
            self.is_validating, True, self.on_ftest
        )

        if not self.on_ftest and self.is_validating is True:
            if self.metric == "cost":
                if self.metric_index == "avg":
                    cost = np.mean (r.cost)
                elif self.metric_index == "sum":
                    cost = np.sum (r.cost)
                else:
                    cost = r.cost [self.metric_index]
            else:
                cost = np.sum (r.cost)
            self.overfitwatch.add_cost (cost, self.is_validating)
        return r

    # batch training  ----------------------------------------------
    def batch (self, x, y, ops = None, **kargs):
        if self.sess is None:
            self.setup ()

        self.is_improved = False
        self.batch_size = x.shape [0]
        # BUG? tf.control_dependencies stuck when using embedding lookup and lstm same time
        _ops = []
        if self.update_ops:
            _ops.append (self.update_ops)
        if isinstance (self.optimize_op, (tuple, list)):
            for op in self.optimize_op:
                _ops.append (op)
        else:
            _ops.append (self.optimize_op)
        if self.summary_op is not None:
            _ops.append (self.summary_op)
        _ops.append (self.logit)
        _ops.append ([cost [1] for cost in self.cost])
        _ops.append (self.learning_rate)

        trailers = 0
        if ops:
            for op in ops:
                _ops.append (op)
            trailers = len (ops)

        r = self.run (*tuple (_ops), x = x, y = y, is_training = True, **kargs)
        clip_all_weights = self.graph.get_collection ("max_norm")
        if clip_all_weights:
            self.sess.run (clip_all_weights)

        if trailers:
            logit, cost, lr = r [:-trailers][-3:]
        else:
            logit, cost, lr = r [-3:]

        global_step = self.get_global_step ()
        if self.summary_op is not None:
            self.writers ["train"].add_summary (r [-(trailers + 4)], global_step)
        return result.Result (
            x, y, logit, cost,
            trailers and r [-trailers:] or None, kargs,
            self.epoch, global_step, self.get_learning_rate (), False, False, False
        )

    # evaluations  ----------------------------------------------
    def resub (self, x, y, ops = [], **kargs):
        self.is_validating = False
        return self.evaluate (x, y, True, ops, **kargs).summary ()
    train = resub

    def valid (self, x, y, ops = [], **kargs):
        self.is_validating = True
        r = self.evaluate (x, y, False, ops, **kargs)
        self.maybe_save_checkpoint (r)
        r.is_improved = self.is_improved
        r.is_overfit = self.is_overfit
        if self.cherry is None or self.is_improved:
            self.set_cherry (r)
        r.summary ()
        self.cherry.summary ()
        return r

    def ptest (self, x, y, ops = [], **kargs):
        self.is_validating = 2
        r = self.evaluate (x, y, False, ops, **kargs)
        r.summary ()
        return r

    def ftest (self, x, y, ops = [], **kargs):
        if self.cost is None:
            with self.graph.as_default ():
                self.add_cost ()
                self.add_performance ()
        self.is_validating = True
        self.on_ftest = True
        return self.evaluate (x, y, False, ops, **kargs).summary ()

    # call consistency -----------------------------------------------
    def improved (self):
        return self.is_improved

    def overfitted (self):
        return self.is_overfit

    # runs for purposes ----------------------------------------------
    def run_for_train (self, *ops, **kargs):
        self.is_validating = False
        return self.run (*ops, is_training = True, **kargs)
    runt = run_for_train

    def run_for_eval (self, *ops, **kargs):
        self.is_validating = True
        return self.run (*ops, is_training = False, **kargs)
    rune = run_for_eval

    def run (self, *ops, **kargs):
        if "seq_length" in kargs and self.seq_length is None:
            raise ValueError ("call DNN.add_seq_length () first")

        if "y" in kargs:
            if "x" in kargs:
                kargs ["y"], kargs ["x"] = self.filter (kargs ["y"], kargs ["x"])
            kargs ["n_sample"] = kargs ["y"].shape [0]
        elif "x" in kargs:
            kargs ["n_sample"] = kargs ["x"].shape [0]

        feed_dict = {}
        for k, v in kargs.items ():
            try:
                attr = getattr (self, k)
            except AttributeError:
                continue
            feed_dict [attr] = v
        result = self.sess.run (ops, feed_dict = feed_dict)
        return result

    # layering -------------------------------------------------------------------
    def dropout (self, layer, dropout = True, activation = None):
        if activation is not None:
           layer = activation (layer)
        if self.in_service or not dropout:
            return layer
        dr = tf.where (tf.less (self.dropout_rate, 0.0), self.random_dropout_rate, self.dropout_rate)
        return tf.layers.dropout (inputs=layer, rate = dr, training = self.is_training)

    def embeddings (self, n_input, size_voca, size_embed, dropout = False):
        weight_init = tf.random_normal_initializer (stddev = (1.0 / size_voca) ** 0.5)
        W = tf.get_variable (
            "embedding",
            [size_voca, size_embed],
            initializer = weight_init
        )
        embed = tf.nn.embedding_lookup (W, tf.cast (n_input, tf.int32))
        if dropout:
            embed = self.dropout (embed)
        return embed

    def lstm (self, *args, **kargs):
        return self._rnn_cell ('LSTMCell', *args, **kargs)

    def gru (self, *args, **kargs):
        return self._rnn_cell ('GRUCell', *args, **kargs)

    def rnn (self, *args, **kargs):
        return self._rnn_cell ('BasicRNNCell', *args, **kargs)

    def _rnn_cell (
            self, cell_name, n_input, hidden_size, lstm_layers = 1,
            activation = None, dynamic = True, dropout = False, kreg = None,
            time_major = False, to_time_major = False
    ):
        try:
            rnn = tf.nn.rnn_cell
            type_rnn = dynamic and tf.nn.dynamic_rnn or tf.nn.static_rnn
        except AttributeError:
            rnn = tf.contrib.rnn
            type_rnn = dynamic and rnn.dynamic_rnn or rnn.static_rnn

        cells = []
        for i in range (lstm_layers):
            lstm = getattr (tf.nn.rnn_cell, cell_name) (hidden_size, activation = activation)
            if dropout:
                keep_prob = 1.0 - self.dropout_rate
                lstm = rnn.DropoutWrapper (lstm, input_keep_prob = keep_prob, output_keep_prob = keep_prob, state_keep_prob = keep_prob)
            cells.append (lstm)

        if  lstm_layers == 1:
            cell = cells [0]
        else:
            cell = rnn.MultiRNNCell (cells)

        # transform time major form
        lstm_in = n_input
        shape = n_input.get_shape()
        dims = len (shape)
        if to_time_major or not dynamic:
            lstm_in = tf.transpose (n_input, [1, 0] + list (range (max (2, dims - 2), dims)))
            time_major = True

        initial_state = cell.zero_state (self.n_sample, tf.float32)
        if dynamic:
            outputs, final_state = type_rnn (cell, lstm_in, time_major = time_major, dtype = tf.float32, initial_state = initial_state, sequence_length = self.seq_length)
        else:
            seq_len = shape [1]
            n_channel = dims >= 3 and shape [2] or 0
            if n_channel:
                lstm_in = tf.reshape (lstm_in, [-1, n_channel])
            lstm_in = tf.layers.dense (lstm_in, hidden_size)
            lstm_in = tf.split (lstm_in, seq_len, 0)
            outputs, final_state = type_rnn (cell, lstm_in, dtype = tf.float32, initial_state = initial_state, sequence_length = self.seq_length)
            if not time_major:
                outputs = tf.transpose (outputs, [1, 0] + list (range (max (2, dims - 2), dims)))
        return outputs, final_state

    def full_connect (self, tensor):
        n_output = 1
        for d in tensor.get_shape ()[1:]:
            n_output *= int (d)
        return tf.reshape (tensor, [-1, n_output])

    def sequencial_connect (self, tensor, seq_len, n_output):
        # outputs is rnn outputs
        fc = self.full_connect (tensor)
        outputs = tf.layers.dense (fc, n_output, activation = None)
        return tf.reshape (outputs, [self.n_sample, seq_len, n_output])

    def batch_norm (self, n_input, activation = None, momentum = 0.99, center = True, scale = True):
        layer = tf.layers.batch_normalization (n_input, momentum = momentum, training = self.is_training, center = center, scale = scale)
        if activation is not None:
           return activation (layer)
        return layer

    def batch_norm_with_dropout (self, n_input, activation = None, momentum = 0.99, center = True, scale = True):
       layer = self.batch_norm (n_input, activation, momentum, center = center, scale = scale)
       return self.dropout (layer)

    def dense (self, n_input, n_output, activation = None, kreg = None):
        return tf.layers.dense (inputs = n_input, units = n_output, activation = activation, kernel_regularizer = kreg)

    def merge (self, *layers):
        return tf.keras.layers.Add ()(list (layers))

    def zero_pad1d (self, input, padding = 1):
        return tf.keras.layers.ZeroPadding1D (padding = padding) (input)

    def zero_pad2d (self, input, padding = (1, 1)):
        return tf.keras.layers.ZeroPadding2D (padding = padding) (input)

    def zero_pad3d (self, input, padding = (1, 1, 1)):
        return tf.keras.layers.ZeroPadding3D (padding = padding) (input)

    def conv1d (self, n_input, filters, kernel = 2, strides = 1, activation = None,  padding = "same", kreg = None):
        return tf.layers.conv1d (inputs = n_input, filters = filters, kernel_size = kernel, strides = strides, padding = padding, activation = activation, kernel_regularizer = kreg)

    def separable_conv1d (self, n_input, filters, kernel = 2, strides = 1, activation = None,  padding = "same", kreg = None):
        return tf.keras.layers.SeparableConv1D (filters, kernel, strides, activation = activation,  padding = padding, kernel_regularizer = kreg) (n_input)

    def max_pool1d (self, n_input, pool = 2, strides = 2, padding = "same"):
        return tf.layers.max_pooling1d (inputs = n_input, pool_size = pool, strides = strides, padding = padding)

    def avg_pool1d (self, n_input, pool = 2, strides = 2, padding = "same"):
        return tf.layers.average_pooling1d (inputs = n_input, pool_size = pool, strides = strides, padding = padding)

    def upsample1d (self, input, size = 2):
        return tf.keras.layers.UpSampling1D (size = size) (input)

    def conv2d (self, n_input, filters, kernel = (2, 2), strides = (1,1), activation = None, padding = "same", kreg = None):
        return tf.layers.conv2d (inputs = n_input, filters = filters, kernel_size = kernel, strides = strides, padding = padding, activation = activation, kernel_regularizer = kreg)

    def separable_conv2d (self, n_input, filters, kernel = (2, 2), strides = (1,1), activation = None, padding = "same", kreg = None):
        return tf.keras.layers.SeparableConv2D (filters, kernel, strides, activation = activation,  padding = padding, kernel_regularizer = kreg) (n_input)

    def max_pool2d (self, n_input, pool = (2, 2), strides = (2, 2), padding = "same"):
        return tf.layers.max_pooling2d (inputs = n_input, pool_size = pool, strides = strides, padding = padding)

    def avg_pool2d (self, n_input, pool = (2, 2), strides = (2, 2), padding = "same"):
        return tf.layers.average_pooling2d (inputs = n_input, pool_size = pool, strides = strides, padding = padding)

    def upsample2d (self, input, size = (2, 2)):
        return tf.keras.layers.UpSampling2D (size = size) (input)

    def conv3d (self, n_input, filters, kernel = (2, 2, 2), strides = (1, 1, 1), activation = None, padding = "same", kreg = None):
        return tf.layers.conv3d (inputs = n_input, filters = filters, kernel_size = kernel, strides = strides, padding = padding, activation = activation, kernel_regularizer = kreg)

    def max_pool3d (self, n_input, pool = (2, 2, 2), strides = (2, 2, 2), padding = "same"):
        return tf.layers.max_pooling3d (inputs = n_input, pool_size = pool, strides = strides, padding = padding)

    def avg_pool3d (self, n_input, pool = (2, 2, 2), strides = (2, 2, 2), padding = "same"):
        return tf.layers.average_pooling3d (inputs = n_input, pool_size = pool, strides = strides, padding = padding)

    def upsample3d (self, input, size = (2, 2, 2)):
        return tf.keras.layers.UpSampling3D (size = size) (input)

    def global_avg_pool1d(self, input):
        return tf.keras.layers.GlobalAveragePooling1D ()(input)

    def global_avg_pool2d(self, input):
        return tf.keras.layers.GlobalAveragePooling2D ()(input)

    def global_avg_pool3d(self, input):
        return tf.keras.layers.GlobalAveragePooling3D ()(input)

    def global_max_pool1d(self, input):
        return tf.keras.layers.GlobalMaxPooling1D ()(input)

    def global_max_pool2d(self, input):
        return tf.keras.layers.GlobalMaxPooling2D ()(input)

    def global_max_pool3d(self, input):
        return tf.keras.layers.GlobalMaxPooling3D ()(input)

    def bernoulli_decode (self, input, n_output):
        y = self.dense (input, n_output, activation = tf.sigmoid)
        return tf.clip_by_value (y, 1e-8, 1 - 1e-8)

    def gaussian_encode (self, input, n_output):
        # https://github.com/hwalsuklee/tensorflow-mnist-VAE/blob/master/vae.py
        gaussian_params = self.dense (input, n_output * 2)
        mean = gaussian_params[:, :n_output]
        stddev = 1e-6 + tf.nn.softplus(gaussian_params[:, n_output:])
        y = mean + stddev * tf.random_normal (tf.shape (mean), 0, 1, dtype=tf.float32)
        return y, mean, stddev

    # helpers ------------------------------------------------------------------
    def l1 (self, scale):
        return tf.contrib.layers.l1_regularizer (scale)

    def l2 (self, scale):
        return tf.contrib.layers.l2_regularizer (scale)

    def l12 (self, scale_l1, scale_l2):
        return tf.contrib.layers.l1_l2_regularizer (scale_l1, scale_l2)

    def max_norm (self, threshold, axes = 1):
        def max_norm_ (weights):
            clipped = tf.clip_by_norm (weights, clip_norm = threshold, axes = axes)
            clip_weights = tf.assign (weights, clipped, name = "max_norm")
            tf.add_to_collection ("max_norm", clip_weights)
            return None
        return max_norm_

    def swish (self, a):
        return tf.nn.sigmoid (a) * a

    def tanh (self, a):
        return 2 * tf.nn.sigmoid (2 * a) - 1

    # optimizer related ------------------------------------------------------
    def scoped_cost (self, cost, scopes = None, task_name = ""):
        if scopes:
            if not isinstance (scopes, (list, tuple)):
                if not task_name:
                    task_name = scopes
                scopes = [scopes]
            t_vars = tf.trainable_variables ()
            collected = set ()
            for var in t_vars:
                for name_ in scopes:
                    if name_ in var.name:
                        collected.add (var)
                        self.__optimzables.add (var)
                        break
        reg_cost = tf.add (cost, self.get_regularization_losses (scopes))
        return cost, reg_cost, list (collected), task_name

    def optimizer (self, name = 'adam', cost = None, learning_rate = None, **karg):
        if cost is None:
            cost = self.cost [0]
        _, reg_cost, var_list, _ = cost
        if var_list:
            karg ["var_list"] = var_list
        if learning_rate is None:
            learning_rate = self.learning_rate
        if self.is_major_optimizer:
            step_var = self.global_step
            self.is_major_optimizer = False
        else:
            step_var = tf.Variable (0, trainable=False)
        return getattr (optimizers, name) (reg_cost, learning_rate, step_var, **karg)

    # override theses ----------------------------------------------------------
    def make_place_holders (self):
        pass

    def make_variables (self):
        pass

    def make_optimizer (self):
        return self.optimizer ("adam")

    def make_logit (self):
        raise NotImplemented

    def make_cost (self):
        raise NotImplemented
        #return tf.constant (0.0)

    def performance_metric (self, r):
        raise NotImplementedError

    # life cycle hooks -------------------------------------------------------------
    def before_graph_create (self):
        pass

    def graph_created (self):
        pass

    def train_setuped (self):
        pass

    # Deprecatings -------------------------------------------------------------
    def trainable (self, *args, **karg):
        warnings.warn (
           "trainable() will be deprecated, use set_learning_rate() and early_terminate()",
            DeprecationWarning
        )
