import numpy as np
cimport numpy as np
from cpython cimport array
import array
cimport libc.math as cmath


def mean_filter(double[:,:,:] data, int window_size): 
    """
    We want to calculate a mean filter with a window function
    and return the mean filtered data. 

    # The input data is 3D
    dimensions(band,channel,time) 

    """

    cdef int nband = data.shape[0]
    cdef int nchan = data.shape[1]
    cdef int ntime = data.shape[2]
    cdef int i,j,k, kk, test_sample
    cdef double local_count

    cdef float[:,:,:] mean_data = np.zeros((nband,nchan,ntime),dtype=np.float64)

    for i in range(nband):
        for j in range(nchan):
            for k in range(ntime):
                local_count = 0
                # First count how many samples in window
                for kk in range(-window_size//2,window_size//2):
                    test_sample = k + kk 
                    if (test_sample >= 0) & (test_sample < ntime):
                        local_count += 1
                # Second calculate the means 
                for kk in range(-window_size//2,window_size//2):
                    test_sample = k + kk 
                    if (test_sample >= 0) & (test_sample < ntime):
                        mean_data[i,j,k] += data[i,j,k+kk]/local_count 

    return mean_data