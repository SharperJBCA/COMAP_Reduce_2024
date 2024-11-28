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
            I = tod[i]
            # Bin to I
            sum_map[pixels[i]] += I*weights[i]
            wei_map[pixels[i]] += weights[i] 
            hit_map[pixels[i]] += 1.0 


def bin_tod_to_rhs(float[:] rhs, float[:] tod, float[:] weights, int offset_length):

    cdef int i,j,k  
    cdef int nsamples = tod.size
    cdef float tod_temp = 0 
    cdef float I
    # First bin the TOD into a map 
    for i in range(nsamples):
            rhs[i//offset_length] += tod[i]*weights[i]

def subtract_map_from_rhs(float[:] rhs, float[:] input_map, int[:] pixels, float[:] weights, int offset_length):

    cdef int i,j,k  
    cdef int nsamples = pixels.size
    cdef int binmax   = input_map.size
    cdef float tod_temp = 0 
    cdef float I
    # First bin the TOD into a map 

    for i in range(nsamples):
        if (pixels[i] >= 0) & (pixels[i] < binmax):
            I = input_map[pixels[i]]
            rhs[i//offset_length] -= I*weights[i]

def bin_map_to_rhs(float[:] rhs, float[:] input_map, float[:] tod, int[:] pixels, float[:] weights, float[:] pa, int offset_length):

    cdef int i,j,k  
    cdef int nsamples = pixels.size
    cdef int binmax   = input_map.size//3
    cdef float tod_temp = 0 
    cdef float I, Q, U
    # First bin the TOD into a map 

    for i in range(nsamples):
        if (pixels[i] >= 0) & (pixels[i] < binmax):
            I = input_map[pixels[i]]
            Q = input_map[pixels[i]+binmax]
            U = input_map[pixels[i]+2*binmax]
            rhs[i//offset_length] += (tod[i] - I)*weights[i]
            tod_temp = (Q*cmath.cos(2*pa[i]) + U*cmath.sin(2*pa[i]))*weights[i+nsamples]
            rhs[i//offset_length + nsamples//offset_length] += (tod[i+nsamples] -  tod_temp) 
            tod_temp = (-Q*cmath.sin(2*pa[i]) + U*cmath.cos(2*pa[i]))*weights[i+2*nsamples]
            rhs[i//offset_length + 2*nsamples//offset_length] += (tod[i+nsamples] -  tod_temp)



def binValues(double[:] image, long[:] pixels, double[:] weights=None, long[:] mask=None):
    """
    A simple binning routine for map-making. Sum is done in place.
    
    Arguments
    image  - 1D array of nypix * nxpix dimensions
    pixels - Indices for 1D image array
    
    Kwargs
    weights - 1D array the same length of pixels.
    mask    - Bool array, skip certain TOD values, 0 = skip, 1 = include
    """

    cdef int i,j,k  
    cdef int nsamples = pixels.size
    cdef int maxbin   = image.size
    for i in range(nsamples):
        if not isinstance(mask, type(None)):
            if mask[i] == 0:
                continue

        if (pixels[i] >= 0) & (pixels[i] < maxbin):
            if isinstance(weights, type(None)):
                image[pixels[i]] += 1.0
            else:#
                image[pixels[i]] += weights[i]


def binValues2Map(double[:] image, long[:] pixels, double[:] weights, long[:] offsetpixels):

    cdef int i
    cdef int nsamples = pixels.size
    cdef int maxbin   = image.size
    cdef int noffsets = weights.size
    for i in range(nsamples):

        if (pixels[i] >= 0) & (pixels[i] < maxbin) & (offsetpixels[i] >= 0) & (offsetpixels[i] < noffsets):
            image[pixels[i]] += weights[offsetpixels[i]]




def EstimateResidual(double[:] residual, 
                     double[:] counts,
                     double[:] offsetval,
                     double[:] offsetwei,
                     double[:] skyval,
                     long[:] offseti, 
                     long[:] pixel):

    cdef int i
    cdef int nsamples = pixel.size
    cdef int maxbin1  = skyval.size
    cdef int noffsets  = residual.size
    
    for i in range(nsamples):

        if ((pixel[i] >= 0) & (pixel[i] < maxbin1)) &\
           ((offseti[i] >= 0) & (offseti[i] < noffsets)) &\
           (offsetwei[i] != 0):
            residual[offseti[i]] += (offsetval[offseti[i]]-skyval[pixel[i]])*offsetwei[i] #offseti[i]]
            counts[offseti[i]] += 1


def EstimateResidualSimplePrior(double[:] output, 
                                double[:] resid_offset,
                                double[:] weights,
                                double[:] resid_sky,
                                long[:] offseti, 
                                long[:] pixel):

    cdef int i
    cdef int nsamples = pixel.size
    cdef int maxbin1  = resid_sky.size
    cdef int noffsets  = output.size
    for i in range(nsamples):

        if ((pixel[i] >= 0) & (pixel[i] < maxbin1)) &\
           ((offseti[i] >= 0) & (offseti[i] < noffsets)) &\
           (weights[offseti[i]] != 0):
            output[offseti[i]] += (resid_offset[offseti[i]]-resid_sky[pixel[i]]) * weights[offseti[i]]

    #for i in range(noffsets):
    #    output[i] += resid_offset[i]

def EstimateResidualFlatMapPrior(double[:] residual, 
                                 double[:] counts,
                                 double[:] offsetval,
                                 double[:] offsetwei,
                                 double[:] skyval,
                                 long[:] offseti, 
                                 long[:] pixel,
                                 double[:] pixhits,
                                 double[:] pcounts):

    cdef int i
    cdef int nsamples = pixel.size
    cdef int maxbin1  = skyval.size
    cdef int noffsets  = residual.size
    
    for i in range(nsamples):

        if ((pixel[i] >= 0) & (pixel[i] < maxbin1)) &\
           ((offseti[i] >= 0) & (offseti[i] < noffsets)) &\
           (offsetwei[i] != 0):
            residual[offseti[i]] += (offsetval[offseti[i]]-skyval[pixel[i]])*offsetwei[i]
            counts[offseti[i]]  += 1

            pcounts[offseti[i]] += 1./pixhits[pixel[i]]**2

    for i in range(noffsets):
        if (pcounts[i] > 0):
            residual[i] += offsetval[i]/pcounts[i]
        else:
            residual[i] += offsetval[i]


def bin_to_ra_az_grid(float[:,:,:] grid, float[:,:,:] weights, float[:,:] tod, int[:] ra_pixels, int[:] az_pixels, float[:] pa):

    cdef int i,j
    cdef int nsamples = ra_pixels.size
    for i in range(nsamples):
        # I 
        grid[0,ra_pixels[i], az_pixels[i]] += tod[0,i]
        weights[0,ra_pixels[i], az_pixels[i]] += 1.0

        # Q 
        grid[1, ra_pixels[i], az_pixels[i]] += tod[1,i]*cmath.cos(2*pa[i]) - tod[2,i]*cmath.sin(2*pa[i])
        weights[1, ra_pixels[i], az_pixels[i]] += 1 

        # U
        grid[2, ra_pixels[i], az_pixels[i]] += tod[1,i]*cmath.sin(2*pa[i]) + tod[2,i]*cmath.cos(2*pa[i])
        weights[2, ra_pixels[i], az_pixels[i]] += 1
