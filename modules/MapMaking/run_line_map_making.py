# run_map_making.py
#
# Description:
#  Main script for calling the MPI destiping code. This is not a module but a script called in CreateMap.py.
#  It can be called totally separately from the command line too as long as the correct config file has been created.
#
import os 

from astropy.io import fits 
import sys 
sys.path.append(os.getcwd())
import logging
from mpi4py import MPI
import argparse
import toml
import numpy as np
from modules.MapMaking.MapMakingModules.ReadLineData import Level2LineDataReader_Nov2024 
from modules.utils import bin_funcs 
from modules.pipeline_control.Pipeline import BadCOMAPFile, update_log_variable,setup_logging

from modules.SQLModule.SQLModule import db, COMAPData, QualityFlag
import matplotlib
matplotlib.use('agg')

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def load_data(parameters):
    """Read in level 2 data for each process."""

    file_list = np.loadtxt(parameters['file_list'], dtype=str,delimiter=',',ndmin=2)
    print(file_list)
    n_files = len(file_list)
    print('NUMBER OF FILES:', n_files)
    n_files_per_process = n_files // size 
    rank_start = rank * n_files_per_process
    rank_end   = (rank + 1) * n_files_per_process if rank < size - 1 else n_files 

    local_data = Level2LineDataReader_Nov2024(tod_data_name=parameters['tod_data_name'])


    local_data.setup_wcs(parameters['wcs_def'], parameters['line_frequency'], parameters['line_segment_width'])

    if local_data.read_files([file_list[i] for i in range(rank_start, rank_end)],
                             database=db,
                             line_frequency=parameters['line_frequency'],
                                line_segment_width=parameters['line_segment_width'],
                             use_flags=False, 
                             feeds=parameters['feeds']):
        return local_data
    else:
        return None

def create_maps(local_data, parameters):
    """Create maps from the offsets."""

    rms_map = np.zeros_like(local_data.sum_map, dtype=np.float32)

    mask = local_data.hits_map > 0 
    lower_hit_bound = np.percentile(local_data.hits_map[mask], 5) 

    mask = local_data.hits_map > lower_hit_bound
    rms_map[:,~mask] = np.nan
    rms_map[:,mask] = np.sqrt(local_data.weight_map[:,mask])  
    

    local_data.sky_map[:,~mask] = np.nan 
    local_data.hits_map[~mask] = np.nan 

    wcs = local_data.wcs  
    naxis3 = local_data.header['NAXIS3'] 
    naxis2 = local_data.header['NAXIS2']
    naxis1 = local_data.header['NAXIS1']

    from astropy.wcs import WCS
    wcs_spec = WCS(naxis=3) 
    wcs_spec.wcs.ctype = ['GLON-CAR', 'GLAT-CAR', 'FREQ']
    wcs_spec.wcs.crval = [local_data.header['CRVAL1'], local_data.header['CRVAL2'], local_data.header['CRVAL3']]
    wcs_spec.wcs.crpix = [local_data.header['CRPIX1'], local_data.header['CRPIX2'], 1]
    wcs_spec.wcs.cdelt = [local_data.header['CDELT1'], local_data.header['CDELT2'], local_data.header['CDELT3']]
    wcs_spec.wcs.cunit = ['','', 'GHz']


    sky_hdu     = fits.PrimaryHDU(local_data.sky_map.reshape(naxis3, naxis2, naxis1), header=wcs_spec.to_header())
    #hits_hdu    = fits.ImageHDU(local_data.hits_map.reshape(naxis2, naxis1), name='HITS', header= wcs.to_header())
    #rms_hdu     = fits.ImageHDU(rms_map.reshape(naxis2, naxis1), name='RMS')

    #for hdu in [sky_hdu, hits_hdu]:
    #    hdu.header.update(wcs.to_header())

    hdul = fits.HDUList([sky_hdu])
    os.makedirs(parameters['output_dir'], exist_ok=True)
    hdul.writeto(f"{parameters['output_dir']}/{parameters['output_filename']}", overwrite=True)


def save_maps(maps, parameters):
    """Save the maps to disk."""

    pass

def main():
    
    parser = argparse.ArgumentParser(description='Run the map making code')
    parser.add_argument('config_file', type=str, help='Path to the configuration file')
    args = parser.parse_args()

    parameters = toml.load(args.config_file) 
    setup_logging(parameters['log_file_name'], prefix_date=False)
    update_log_variable('Rank: {:02d}'.format(rank))
    db.connect(parameters['database'])

    local_data = load_data(parameters) 

    if local_data is None:
        return

    if rank == 0:
        maps = create_maps(local_data, parameters) 

   #save_maps(maps, parameters) 

if __name__ == "__main__":
    main()  