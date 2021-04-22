`get_data.py` downloads all the LoTSS and Planck lensing data
`run_pointing_maps.py` generates per-pointing healpix maps containing information about local rms. It makes use of `make_pointing_map.py` to run over batches of pointings per job.
`join_pointing_maps.py` merges all per-pointing maps and generates a geometric mask and an RMS map over the full footprint.
`get_p_map.py` generates a "probability" or "mean density" map from the RMS map over the full footprint.
`compute_Cl.py` computes the power spectra given all of the above

MISSING:
- Rotate kappa


To run the full vanilla pipeline from scratch you would run something like:
```
> echo "Downloading data"
> python get_data.py
> echo "Generating per-pointing maps"
> python run_pointing_maps.py <output directory>
> echo "Merging per-pointing maps"
> python join_pointing_maps.py <output directory>
> python get_p_map.py <output directory>/maps_all.fits.gz flux_threshold SN_threshold <output directory>/p_map.fits
> echo "Computing power spectra"
> python compute_Cl.py <output cl directory> <catalog name> <output directory>/p_map.fits <kappa map name> <kappa mask name> <noise kappa name> flux_threshold SN_threshold
```
