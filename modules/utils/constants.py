# COMAP instrument constants
#
# Centralized definitions for hardware parameters used across the pipeline.
# Import from here instead of hardcoding in individual modules.

NFEEDS = 19          # Number of science feeds (pixels 1-19)
NBANDS = 4           # Number of frequency bands per feed
NCHANNELS = 1024     # Number of spectral channels per band (before binning)
GROUND_FEED = 20     # Feed number for the ground-pickup pixel (always skipped)
