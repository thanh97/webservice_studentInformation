import cv2
import numpy as np

def average_hash (arr, size = 16):
    geryscale = cv2.resize (arr, (size, size))
    if geryscale.shape [-1] > 1:
        geryscale = cv2.cvtColor(geryscale, cv2.COLOR_BGR2GRAY)
    return 1 * (geryscale > geryscale.mean ())
    
def hamming_dist (a, b):
    return (a != b).sum () / np.product (a.shape)


        