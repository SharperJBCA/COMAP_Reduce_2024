"""
Destriper.py -- An MPI ready implementation of the Destriping algorithm.

Includes a test script + some methods simulating noise and signal

run Destriper.test() to run example script.

Requires a wrapper that will creating the pointing, weights and tod vectors
that are needed to pass to the Destriper.

This implementation does not care about the coordinate system

Refs:
Sutton et al. 2011 

"""
import matplotlib 
matplotlib.use('Agg')
import numpy as np
from matplotlib import pyplot
from scipy.sparse.linalg import LinearOperator
from scipy.ndimage import gaussian_filter
from modules.utils import bin_funcs
import healpy as hp
import logging
import numpy as np
import os

from modules.MapMaking.MapMakingModules.ReadData import Level2DataReader_Nov2024

import numpy as np
from scipy.sparse import csr_matrix

def create_pointing_matrix(pixels, n_pixels):
    """
    Convert 1D pixel indices into sparse pointing matrix P.
    
    Args:
        pixels: 1D array of pixel indices for each time sample
        n_pixels: Total number of pixels in map
        
    Returns:
        P: Sparse matrix of shape (n_samples, n_pixels)
    """
    n_samples = len(pixels)
    rows = np.arange(n_samples)
    cols = pixels
    data = np.ones_like(pixels, dtype=float)
    
    # Filter invalid pixels
    mask = (pixels >= 0) & (pixels < n_pixels)
    
    return csr_matrix((data[mask], (rows[mask], cols[mask])), 
                     shape=(n_samples, n_pixels))

def create_offset_matrix(n_samples, offset_length):
    """
    Create sparse offset mapping matrix F where F[t,α] = 1 if time t is in offset chunk α
    
    Args:
        n_samples: Total number of time samples
        offset_length: Length of each offset chunk
        
    Returns:
        F: Sparse matrix of shape (n_samples, n_offsets)
    """
    n_offsets = n_samples // offset_length
    rows = np.arange(n_samples)
    cols = rows // offset_length
    data = np.ones(n_samples)
    
    return csr_matrix((data, (rows, cols)), 
                     shape=(n_samples, n_offsets))

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size() 

def sum_map_all_inplace(m): 
    """Sum array elements over all MPI processes"""
    if m.dtype == np.float64:
        mpi_type = MPI.DOUBLE
    else:
        mpi_type = MPI.FLOAT
    comm.Allreduce(MPI.IN_PLACE,
        [m, mpi_type],
        op=MPI.SUM
        )
    return m 

def mpi_sum(x):
    """Sum all sums over all MPI processes"""
    # Sum the local values
    local = np.array([np.sum(x)])
    comm.Allreduce(MPI.IN_PLACE, local, op=MPI.SUM)
    return local[0]

class AxCOMAP:
    def __init__(self,data_object : Level2DataReader_Nov2024):
        self.data_object = data_object

        self.Ax      = np.zeros_like(data_object.rhs, dtype=np.float32)
        self.sky_map = np.zeros_like(data_object.sum_map, dtype=np.float32)
        self.sum_map = np.zeros_like(data_object.sum_map, dtype=np.float32)
        self.wei_map = np.zeros_like(data_object.weight_map, dtype=np.float32)

    def __call__(self,offsets,extend=True): 
        """
        """
        if offsets.dtype != np.float32:
            offsets = offsets.astype(np.float32)

        self.sum_map *= 0
        self.wei_map *= 0 
        self.sky_map *= 0
        self.Ax *= 0
        
        bin_funcs.bin_offset_to_map(offsets,
                                    self.sky_map,
                                    self.sum_map,
                                    self.wei_map,
                                    self.data_object.pixels,
                                    self.data_object.weights,
                                    self.data_object.offset_length)
                
        self.sum_map = sum_map_all_inplace(self.sum_map)
        self.wei_map = sum_map_all_inplace(self.wei_map)
        mask = self.wei_map > 0 
        self.sky_map[mask] = self.sum_map[mask]/self.wei_map[mask] 

        # Sum up the TOD into the offsets
        bin_funcs.bin_offset_to_rhs(self.Ax,
                                 offsets, 
                                 self.sky_map,
                                 self.data_object.pixels,
                                 self.data_object.weights, 
                                 self.data_object.offset_length)         

        return self.Ax #+ offsets 
        # +_tod is like adding prior that sum(offsets) = 0. (F^T N^-1 Z F + 1)a = F^T N^-1 d 

def cgm( Ax, 
        threshold : float = 1e-3, 
        niter : int =1000, 
        verbose : bool = False, 
        x0=None):
    """
    Biconjugate CGM implementation from Numerical Recipes 2nd, pg 83-85
    
    arguments:
    b - Array
    Ax - Function that applies the matrix A
    
    kwargs:
    
    Notes:
    1) Ax can be a class with a __call__ function defined to act like a function
    2) Weights should be average weight for an offset, e.g. w_b = sum(w)/offset_length
    3) Need to add a preconditionor matrix step
    """
    
    Ax(np.zeros(Ax.data_object.n_offsets, dtype=np.float32))

    A = LinearOperator((Ax.data_object.n_offsets, Ax.data_object.n_offsets), matvec = Ax)


    if isinstance(x0,type(None)):
        x0 = np.zeros(Ax.data_object.n_offsets, dtype=np.float32)
        

    r  = Ax.data_object.rhs - A.matvec(x0)
    rb = Ax.data_object.rhs - A.matvec(x0)
    p  = r*1.
    pb = rb*1.

    mask = Ax.data_object.weight_map > 0 
    naive_sky = np.zeros_like(Ax.data_object.sum_map)
    naive_sky[mask] = Ax.data_object.sum_map[mask]/Ax.data_object.weight_map[mask]

    thresh0 = mpi_sum(r*rb) 
    for i in range(niter):
        q = A.matvec(pb)


        rTrb = mpi_sum(r*rb) 
        alpha= rTrb/mpi_sum(pb*q)

        x0 += alpha*pb

        r = r - alpha*A.matvec(p)
        rb= rb- alpha*A.matvec(pb)
        
        beta = mpi_sum(r*rb)/rTrb

        
        p = r + beta*p
        pb= rb+ beta*pb
        

        delta = mpi_sum(r*rb)/thresh0
        
        if rank == 0:
            print(f'Iter: {i} Delta: {delta}')
            logging.info(f'Iter: {i} Delta: {delta}')
            #pyplot.plot(naive_sky[Ax.data_object.pixels])
            #pyplot.plot(np.repeat(x0,Ax.data_object.offset_length))
            #pyplot.savefig(f'iter_{i}.png')
            #pyplot.close()
        if verbose:
            print(delta)
        if delta < threshold:
            break

        sum_map = np.zeros_like(Ax.data_object.sum_map)
        wei_map = np.zeros_like(Ax.data_object.weight_map)
        sky_map = np.zeros_like(Ax.data_object.sum_map)
        bin_funcs.bin_offset_to_map(x0, sky_map, 
                                    sum_map, 
                                    wei_map, 
                                    Ax.data_object.pixels, Ax.data_object.weights, Ax.data_object.offset_length)
                                            

    sum_map    = np.zeros_like(Ax.data_object.sum_map, dtype=np.float32)
    wei_map    = np.zeros_like(Ax.data_object.weight_map, dtype=np.float32)
    offset_map = np.zeros_like(Ax.data_object.sum_map, dtype=np.float32)
    bin_funcs.bin_offset_to_map(x0,
                                offset_map,
                                sum_map,
                                wei_map,
                                Ax.data_object.pixels,
                                Ax.data_object.weights,
                                Ax.data_object.offset_length)
    
    sum_map = sum_map_all_inplace(sum_map)
    wei_map = sum_map_all_inplace(wei_map)
    mask = wei_map > 0 
    offset_map[mask] = sum_map[mask]/wei_map[mask]

    return offset_map 

