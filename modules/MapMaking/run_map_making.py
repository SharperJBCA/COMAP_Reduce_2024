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
from modules.MapMaking.MapMakingModules.ReadData import Level2DataReader
from modules.MapMaking.MapMakingModules.Destriper import AxCOMAP, cgm
from modules.utils import bin_funcs 
from modules.pipeline_control.Pipeline import BadCOMAPFile, update_log_variable,setup_logging

from modules.SQLModule.SQLModule import SQLModule, COMAPData, QualityFlag

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def load_data(parameters):
    """Read in level 2 data for each process."""

    file_list = np.loadtxt(parameters['file_list'], dtype=str,ndmin=1)
    n_files = len(file_list)
    print('NUMBER OF FILES:', n_files)
    n_files_per_process = n_files // size 
    rank_start = rank * n_files_per_process
    rank_end   = (rank + 1) * n_files_per_process if rank < size - 1 else n_files 

    if parameters['database'] is not None:
        db = SQLModule() 
        db.connect(parameters['database'])
    else:
        db = None

    local_data = Level2DataReader(tod_dataset=parameters['tod_data_name'], 
                                          offset_length=parameters['offset_length'], 
                                          planck_30_path=parameters['planck_30_path'],
                                          calib_path=parameters['calib_path'], 
                                          sigma_red_cutoff=parameters['sigma_red_cutoff'],
                                          database=db,
                                          jackknife=parameters.get('jackknife_odd_even',None),
                                          band_index=parameters['band'])
    db.disconnect()
    band, channel = np.unravel_index(parameters['band'], (4,2))

    local_data.setup_wcs(parameters['wcs_def'])

    if local_data.read_files([file_list[i] for i in range(rank_start, rank_end)],
                             use_flags=False, 
                             feeds=parameters['feeds']):
        return local_data
    else:
        return None

def run_destriper(local_data, parameters):
    """Run destriper on each process."""

    Ax = AxCOMAP(local_data,
            lambda_ridge = parameters.get("lambda_ridge", 0.0),
            alpha_grad   = parameters.get("alpha_grad",   0.0),
            alpha_curv   = parameters.get("alpha_curv",   0.0))
 

    offsets = cgm(Ax, threshold=parameters.get('threshold', 1e-6)) 

    return offsets


def create_maps(
    offset_map,
    local_data,
    parameters,
    *,
    hit_percentile=5.0,                 # drop the lowest X% of hit counts
    min_hits_floor=1,                   # never accept < this many hits
    edge_width=2,                       # border width (pixels) to scrutinize
    edge_clip_percentiles=(0.5, 99.5),  # only clip border values outside these (from interior)
    add_mask_hdu=True,
):
    """
    Build destriped / RMS / HITS / NAIVE maps, apply masks, and write FITS with nice headers.

    Bit-mask (uint8) semantics if written:
      bit 0 (1): pixel was never hit
      bit 1 (2): pixel below hit-percentile threshold
      bit 2 (4): border pixel clipped as outlier
    """
    # --- pull arrays (1D) ---
    sum_map    = local_data.data.sum_map.astype(np.float32, copy=False)
    weight_map = local_data.data.weight_map.astype(np.float32, copy=False)
    hits_map   = local_data.data.hits_map.astype(np.float32, copy=False)
    sky_map    = local_data.data.sky_map.astype(np.float32, copy=False)
    off_map    = offset_map.astype(np.float32, copy=False)

    # shapes / WCS
    wcs   = local_data.wcs
    ny    = int(local_data.header["NAXIS2"])
    nx    = int(local_data.header["NAXIS1"])
    npix  = nx * ny
    assert sum_map.size == npix, "Map size mismatch with WCS header."

    # --- core products (1D) ---
    destriped = np.full_like(sky_map, np.nan, dtype=np.float32)
    hit_any   = hits_map > 0
    destriped[hit_any] = sky_map[hit_any] - off_map[hit_any]

    rms = np.full_like(sky_map, np.nan, dtype=np.float32)
    good_w = (weight_map > 0) & hit_any
    rms[good_w] = np.sqrt(1.0 / weight_map[good_w])

    # --- low-hit masking ---
    if np.any(hit_any):
        lower_hit_bound = np.percentile(hits_map[hit_any], hit_percentile)
    else:
        lower_hit_bound = np.inf
    hit_thresh = np.maximum(lower_hit_bound, float(min_hits_floor))
    mask_lowhits = hits_map < hit_thresh
    mask_unhit   = ~hit_any

    # --- border outlier clipping (on destriped map) ---
    d2 = destriped.reshape(ny, nx)
    h2 = hits_map.reshape(ny, nx)

    # define fixed-width border
    border = np.zeros((ny, nx), dtype=bool)
    if edge_width > 0:
        border[:edge_width, :]  = True
        border[-edge_width:, :] = True
        border[:, :edge_width]  = True
        border[:, -edge_width:] = True

    # interior is where we compute robust value range
    interior = (~border) & (h2 > 0)
    if np.any(interior):
        lo, hi = np.nanpercentile(d2[interior], edge_clip_percentiles)
        border_valid = border & (h2 > 0)
        border_outlier = border_valid & ((d2 < lo) | (d2 > hi))
    else:
        border_outlier = np.zeros_like(border, dtype=bool)

    # --- combine final mask ---
    # keep pixels that: have hits, pass hit threshold, and are not clipped on border
    keep = (~mask_unhit) & (~mask_lowhits)
    keep2d = keep.reshape(ny, nx) & (~border_outlier)

    # apply mask
    destriped[~keep2d.ravel()] = np.nan
    sky_map[~keep2d.ravel()]   = np.nan
    hits_map[~keep2d.ravel()]  = np.nan
    rms[~keep2d.ravel()]       = np.nan

    # --- build HDUs (2D) ---
    H  = wcs.to_header()
    pri = fits.PrimaryHDU(destriped.reshape(ny, nx))
    h_naive = fits.ImageHDU(sky_map.reshape(ny, nx),  name="NAIVE")
    h_hits  = fits.ImageHDU(hits_map.reshape(ny, nx),  name="HITS")
    h_rms   = fits.ImageHDU(rms.reshape(ny, nx),       name="RMS")

    for hdu in (pri, h_naive, h_hits, h_rms):
        hdu.header.update(H)

    # --- optional mask HDU ---
    hdus = [pri, h_naive, h_hits, h_rms]
    if add_mask_hdu:
        mask_bits = np.zeros(npix, dtype=np.uint8)
        mask_bits[mask_unhit]   |= 1
        mask_bits[mask_lowhits] |= 2
        mask_bits[border_outlier.ravel()] |= 4
        h_mask = fits.ImageHDU(mask_bits.reshape(ny, nx), name="MASK")
        h_mask.header.update(H)
        hdus.append(h_mask)

    # --- headers: provenance, knobs, basics ---
    def _provenance(h):
        # units / basics
        h["BUNIT"]   = parameters.get("bunit", "K_RJ")
        h["EXTNAME"] = h.get("EXTNAME", "PRIMARY")
        # pipeline
        h["HIERARCH PIPE.NAME"]   = "COMAP Destriper"
        h["HIERARCH PIPE.VERS"]   = parameters.get("pipe_version", "2024-11")
        h["HIERARCH PIPE.OFFLEN"] = int(local_data.data.offset_length)
        h["HIERARCH PIPE.HITPCT"] = float(hit_percentile)
        h["HIERARCH PIPE.HITMIN"] = int(min_hits_floor)
        h["HIERARCH PIPE.EDGEW"]  = int(edge_width)
        h["HIERARCH PIPE.CLIPLO"] = float(edge_clip_percentiles[0])
        h["HIERARCH PIPE.CLIPHI"] = float(edge_clip_percentiles[1])
        # solver regularisation (Planck)
        h["HIERARCH REG.PLANCKCUT"]  = float(getattr(local_data, "planck_quiet_cut_soft", np.nan))
        h["HIERARCH REG.PLANCKDW"]   = float(getattr(local_data, "planck_downweight", np.nan))
        # band/channel/feeds
        h["HIERARCH DATA.BANDIDX"] = int(getattr(local_data, "band_index", -1))
        #h["HIERARCH DATA.FEEDS"]   = str(parameters.get("feeds", "all"))
        # history trail
        h.add_history("Destriped map = sky - offset_map")
        h.add_history(f"Masked low-hits below {hit_percentile:.1f}th percentile (>= {hit_thresh:.1f})")
        h.add_history(f"Clipped border outliers using interior percentiles {edge_clip_percentiles}")

    for hdu in hdus:
        _provenance(hdu.header)

    # helpful DATAMIN/DATAMAX (ignore NaN)
    def _set_minmax(hdu):
        data = hdu.data
        if data is None:
            return
        finite = np.isfinite(data)
        if np.any(finite):
            hdu.header["DATAMIN"] = float(np.nanmin(data[finite]))
            hdu.header["DATAMAX"] = float(np.nanmax(data[finite]))
    for hdu in hdus:
        _set_minmax(hdu)

    # --- write ---
    hdul = fits.HDUList(hdus)
    os.makedirs(parameters["output_dir"], exist_ok=True)
    out = os.path.join(parameters["output_dir"], parameters["output_filename"])
    hdul.writeto(out, overwrite=True)

def create_maps_old(offset_map, local_data, parameters):
    """Create maps from the offsets."""

    rms_map = np.zeros_like(local_data.data.sum_map, dtype=np.float32)
    destriped_map = np.zeros_like(local_data.data.sum_map, dtype=np.float32) 

    mask = local_data.data.hits_map > 0 
    destriped_map[mask] = local_data.data.sky_map[mask] - offset_map[mask] 

    lower_hit_bound = np.percentile(local_data.data.hits_map[mask], 5) 

    mask = local_data.data.hits_map > lower_hit_bound
    rms_map[~mask] = np.nan
    rms_map[mask] = np.sqrt(1./local_data.data.weight_map[mask])  
    

    destriped_map[~mask] = np.nan
    local_data.data.sky_map[~mask] = np.nan 
    local_data.data.hits_map[~mask] = np.nan 

    wcs = local_data.wcs  
    naxis2 = local_data.header['NAXIS2']
    naxis1 = local_data.header['NAXIS1']


    primary_hdu = fits.PrimaryHDU(destriped_map.reshape(naxis2, naxis1))
    sky_hdu     = fits.ImageHDU(local_data.data.sky_map.reshape(naxis2, naxis1), name='NAIVE')
    hits_hdu    = fits.ImageHDU(local_data.data.hits_map.reshape(naxis2, naxis1), name='HITS')
    rms_hdu     = fits.ImageHDU(rms_map.reshape(naxis2, naxis1), name='RMS')

    for hdu in [primary_hdu, sky_hdu, hits_hdu, rms_hdu]:
        hdu.header.update(wcs.to_header())

    hdul = fits.HDUList([primary_hdu, sky_hdu, hits_hdu, rms_hdu])
    os.makedirs(parameters['output_dir'], exist_ok=True)
    hdul.writeto(f"{parameters['output_dir']}/{parameters['output_filename']}", overwrite=True)



def main():
    
    parser = argparse.ArgumentParser(description='Run the map making code')
    parser.add_argument('config_file', type=str, help='Path to the configuration file')
    args = parser.parse_args()

    parameters = toml.load(args.config_file) 
    setup_logging(parameters['log_file_name'], prefix_date=False)
    update_log_variable('Rank: {:02d}'.format(rank))

    local_data = load_data(parameters) 

    if local_data is None:
        return

    offset_map = run_destriper(local_data, parameters) 

    if rank == 0:
        maps = create_maps(offset_map, local_data, parameters) 

if __name__ == "__main__":
    main()  