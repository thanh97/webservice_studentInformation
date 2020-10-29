import numpy as np

def _get_scaling_range (normrange):
    if isinstance (normrange, (list, tuple)):
        scale_min, scale_max = normrange
    else:
        scale_min, scale_max = -1, 1
    return scale_min, scale_max    

def standardize (x, mean, std):
    return (x - mean) / std
    
def scaling (x, min_, gap, normrange):
    scale_min, scale_max = _get_scaling_range (normrange) 
    return np.clip (scale_min + (scale_max - scale_min) * ((x - min_) / gap), scale_min, scale_max)
    
def normalize (x, *args):
    pca = None
    if isinstance (x, list): 
        x = np.array (x)    
    if len (args) == 8:
        # old version
        mean, std, min_, gap, pca_k, pca, _normalize, _standardize = args        
    else:
        mean, std, min_, gap, pca_k, eigen_vecs, pca_mean, _normalize, _standardize = args
        
    if _standardize: # 0 mean, 1 var
        x = standardize (x, mean, std)
    if _normalize: # -1 to 1
        x = scaling (x, min_, gap, normalize)
        
    if pca_k: # PCA
        orig_shape = x.shape
        if len (orig_shape) == 3:
            x = x.reshape ([orig_shape [0]  * orig_shape [1], orig_shape [2]])
        if pca:
            # for old version
            x = pca.transform (x)
        else:    
            x = np.dot (x - pca_mean, eigen_vecs)
        if len (orig_shape) == 3:
            x = x.reshape ([orig_shape [0], orig_shape [1], pca_k])    

    return x
     