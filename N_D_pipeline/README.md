`get_data.py` downloads all the LoTSS and Planck lensing data
`run_pointing_maps.py` generates per-pointing healpix maps containing information about local rms. It makes use of `make_pointing_map.py` to run over batches of pointings per job.
`join_pointing_maps.py` merges all per-pointing maps and generates a geometric mask and an RMS map over the full footprint.
`get_p_map.py` generates a "probability" or "mean density" map from the RMS map over the full footprint.
