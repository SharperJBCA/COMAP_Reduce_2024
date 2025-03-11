import numpy as np
cimport numpy as np
from cpython cimport array
import array
cimport libc.math as cmath

def bin_tod_to_map(float[:] sum_map, float[:] wei_map, float[:] hit_map, float[:] tod, float[:] weights, int[:] pixels):

    cdef int i,j,k  
    cdef int nsamples = pixels.size
    cdef int binmax   = sum_map.size
    cdef float I

    for i in range(nsamples):
        if (pixels[i] >= 0) & (pixels[i] < binmax):
            # Bin to I
            sum_map[pixels[i]] += tod[i]*weights[i]
            wei_map[pixels[i]] += weights[i] 
            hit_map[pixels[i]] += 1.0 


def bin_tod_to_rhs(float[:] rhs, float[:] tod,  float[:] weights,  int offset_length):

    cdef int i
    cdef int nsamples = tod.size
    # First bin the TOD into a map 
    for i in range(nsamples):
            rhs[i//offset_length] += tod[i]*weights[i]

def subtract_sky_map_from_rhs(float[:] rhs,  float[:] weights, float[:] sky_map, int[:] pixels, int offset_length):
    
    cdef int i
    cdef int nsamples = pixels.size
    cdef int npixels = sky_map.size
    for i in range(nsamples):
        if (pixels[i] >= 0) & (pixels[i] < npixels):
            rhs[i//offset_length] -= sky_map[pixels[i]]*weights[i]


def bin_offset_to_rhs(float[:] residuals, 
                      float[:] offsets, 
                      float[:] sky_map, 
                      int[:] pixels, 
                      float[:] weights,  
                      int offset_length):

    cdef int i
    cdef int nsamples = pixels.size
    cdef int npixels = sky_map.size
    for i in range(nsamples):
        if (pixels[i] >= 0) & (pixels[i] < npixels) & (weights[i]  != 0):
            residuals[i//offset_length] += weights[i]*(offsets[i//offset_length] - sky_map[pixels[i]])

def bin_offset_to_map(float[:] offsets, 
                      float[:] sky_map, 
                      float[:] sum_map,
                      float[:] wei_map,
                      int[:]   pixels, 
                      float[:] weights,  
                      int offset_length):

    cdef int i
    cdef int nsamples = pixels.size
    cdef int npixels  = sum_map.size
    for i in range(nsamples):
        if (pixels[i] >= 0) & (pixels[i] < npixels) & (weights[i]  != 0):
            sum_map[pixels[i]] += offsets[i//offset_length]*weights[i]
            wei_map[pixels[i]] += weights[i]

    for i in range(npixels):
        if wei_map[i] > 0:
            sky_map[i] = sum_map[i]/wei_map[i]


