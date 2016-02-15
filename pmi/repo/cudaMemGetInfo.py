# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 19:32:03 2015
https://gist.github.com/justinfx/10458017
@author: lexx
"""

import ctypes

# Path to location of libcudart
_CUDA = "/Developer/NVIDIA/CUDA-7.5/lib/libcudart.7.5.dylib"
cuda = ctypes.cdll.LoadLibrary(_CUDA)

cuda.cudaMemGetInfo.restype = int
cuda.cudaMemGetInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

cuda.cudaGetErrorString.restype = ctypes.c_char_p
cuda.cudaGetErrorString.argtypes = [ctypes.c_int]

def cudaMemGetInfo(mb=True):
    """
    Return (free, total) memory stats for CUDA GPU
    Default units are bytes. If mb==True, return units in MB
    """
    free = ctypes.c_size_t()
    total = ctypes.c_size_t()
    ret = cuda.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total))

    if ret != 0:
        err = cuda.cudaGetErrorString(ret)
        raise RuntimeError("CUDA Error (%d): %s" % (ret, err))

    if mb:
        scale = 1024.0**2
        return free.value / scale, total.value / scale
    else:
        return free.value, total.value
if __name__ == ‘__main__’:
    cudaMemGetInfo(mb=True)