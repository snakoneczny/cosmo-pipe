#!/bin/bash

dirDR1="/mnt/extraspace/damonge/LensLotss/DR1_data"
dirDR2="/mnt/extraspace/damonge/LensLotss/DR2_data"
dirMV="/mnt/extraspace/damonge/LensLotss/MV"
run_get_data=false
run_pointings=false
run_join_pointings=false
run_get_p_maps=false
run_get_p_maps_from_catalog=false
run_DR1_cls=false
run_DR2_cls=false


# Download all data
if [ $run_get_data = true ] ; then
    echo "Downloading all data"
    python3 get_data.py
    echo " "
fi

# Project pointings into healpix
# Both DR1 and DR2
if [ $run_pointings = true ] ; then
    echo "Projecting pointings"
    python3 run_pointings.py
    echo " "
fi

# Join pointing maps
# Both DR1 and DR2
if [ $run_join_pointings = true ] ; then
    echo "Joining pointings"
    python3 join_pointing_maps.py
    echo " "
fi

# Get probability maps
if [ $run_get_p_maps = true ] ; then
    echo "DR2 p-map"
    python3 get_p_map.py ${dirDR2}/maps_all.fits.gz 2. 5. ${dirDR2}/p_map_Icut2p0_SN5p0.fits.gz
    echo "DR1 p-map"
    python3 get_p_map.py ${dirDR1}/maps_all.fits.gz 2. 5. ${dirDR1}/p_map_Icut2p0_SN5p0_allpointings.fits.gz
    echo "DR1 p-map (good pointings only)"
    python3 get_p_map.py ${dirDR1}/maps_good.fits.gz 2. 5. ${dirDR1}/p_map_Icut2p0_SN5p0_goodpointings.fits.gz
    echo " "
fi

# Probability maps from catalogs
if [ $run_get_p_maps_from_catalog = true ] ; then
    echo "DR2 p-map"
    python3 get_p_map_from_catalog.py ${dirDR2}/LoTSS_DR2_v100.srl.fits ${dirDR2}/maps_all.fits.gz 2. 5. ${dirDR2}/p_map_Icut2p0_SN5p0_fromcat.fits.gz
    echo "DR1 p-map"
    python3 get_p_map_from_catalog.py ${dirDR1}/radio_catalog.fits ${dirDR1}/maps_all.fits.gz 2. 5. ${dirDR1}/p_map_Icut2p0_SN5p0_allpointings_fromcat.fits.gz
    echo "DR1 p-map (good pointings only)"
    python3 get_p_map_from_catalog.py ${dirDR1}/radio_catalog.fits ${dirDR1}/maps_good.fits.gz 2. 5. ${dirDR1}/p_map_Icut2p0_SN5p0_goodpointings_fromcat.fits.gz
    echo " "
fi

# DR1 power spectra
if [ $run_DR1_cls = true ] ; then
    echo "Cls, DR1, all pointings, radio catalog"
    addqueue -c cls_DR1_all_radiocat -n 1x28 -s -q berg -m 2 /usr/bin/python3 compute_Cl.py ${dirDR1}/cls_Icut2p0_SN5p0_all ${dirDR1}/radio_catalog.fits ${dirDR1}/p_map_Icut2p0_SN5p0_allpointings.fits.gz ${dirMV}/kmap_rotated.fits ${dirMV}/mask_rot_apo0.200_C1.fits.gz ${dirMV}/nlkk.dat radiocat 2. 5.
    echo "Cls, DR1, good pointings, radio catalog"
    addqueue -c cls_DR1_good_radiocat -n 1x28 -s -q berg -m 2 /usr/bin/python3 compute_Cl.py ${dirDR1}/cls_Icut2p0_SN5p0_good ${dirDR1}/radio_catalog.fits ${dirDR1}/p_map_Icut2p0_SN5p0_goodpointings.fits.gz ${dirMV}/kmap_rotated.fits ${dirMV}/mask_rot_apo0.200_C1.fits.gz ${dirMV}/nlkk.dat radiocat 2. 5.
    echo "Cls, DR1, all pointings, VAC"
    addqueue -c cls_DR1_all_vac -n 1x12 -s -q cmb -m 2 /usr/bin/python3 compute_Cl.py ${dirDR1}/cls_Icut2p0_SN5p0_all ${dirDR1}/hetdex_optical_ids.fits ${dirDR1}/p_map_Icut2p0_SN5p0_allpointings.fits.gz ${dirMV}/kmap_rotated.fits ${dirMV}/mask_rot_apo0.200_C1.fits.gz ${dirMV}/nlkk.dat VAC 2. 5.
    echo "Cls, DR1, good pointings, VAC"
    addqueue -c cls_DR1_good_vac -n 1x12 -s -q cmb -m 2 /usr/bin/python3 compute_Cl.py ${dirDR1}/cls_Icut2p0_SN5p0_good ${dirDR1}/hetdex_optical_ids.fits ${dirDR1}/p_map_Icut2p0_SN5p0_goodpointings.fits.gz ${dirMV}/kmap_rotated.fits ${dirMV}/mask_rot_apo0.200_C1.fits.gz ${dirMV}/nlkk.dat VAC 2. 5.
    echo " "

    echo "Cls, DR1, all pointings, radio catalog, p from catalog"
    addqueue -c cls_DR1_all_radiocat_pcat -n 1x28 -s -q berg -m 2 /usr/bin/python3 compute_Cl.py ${dirDR1}/cls_Icut2p0_SN5p0_all_pcat ${dirDR1}/radio_catalog.fits ${dirDR1}/p_map_Icut2p0_SN5p0_allpointings_fromcat.fits.gz ${dirMV}/kmap_rotated.fits ${dirMV}/mask_rot_apo0.200_C1.fits.gz ${dirMV}/nlkk.dat radiocat 2. 5.
    echo "Cls, DR1, good pointings, radio catalog, p from catalog"
    addqueue -c cls_DR1_good_radiocat_pcat -n 1x28 -s -q berg -m 2 /usr/bin/python3 compute_Cl.py ${dirDR1}/cls_Icut2p0_SN5p0_good_pcat ${dirDR1}/radio_catalog.fits ${dirDR1}/p_map_Icut2p0_SN5p0_goodpointings_fromcat.fits.gz ${dirMV}/kmap_rotated.fits ${dirMV}/mask_rot_apo0.200_C1.fits.gz ${dirMV}/nlkk.dat radiocat 2. 5.
    echo "Cls, DR1, all pointings, VAC, p from catalog"
    addqueue -c cls_DR1_all_vac_pcat -n 1x12 -s -q cmb -m 2 /usr/bin/python3 compute_Cl.py ${dirDR1}/cls_Icut2p0_SN5p0_all_pcat ${dirDR1}/hetdex_optical_ids.fits ${dirDR1}/p_map_Icut2p0_SN5p0_allpointings_fromcat.fits.gz ${dirMV}/kmap_rotated.fits ${dirMV}/mask_rot_apo0.200_C1.fits.gz ${dirMV}/nlkk.dat VAC 2. 5.
    echo "Cls, DR1, good pointings, VAC, p from catalog"
    addqueue -c cls_DR1_good_vac_pcat -n 1x12 -s -q cmb -m 2 /usr/bin/python3 compute_Cl.py ${dirDR1}/cls_Icut2p0_SN5p0_good_pcat ${dirDR1}/hetdex_optical_ids.fits ${dirDR1}/p_map_Icut2p0_SN5p0_goodpointings_fromcat.fits.gz ${dirMV}/kmap_rotated.fits ${dirMV}/mask_rot_apo0.200_C1.fits.gz ${dirMV}/nlkk.dat VAC 2. 5.
    echo " "
fi

# DR1 power spectra
if [ $run_DR1_cls = true ] ; then
    echo "Cls, DR2"
    addqueue -c cls_DR2 -n 1x28 -s -q berg -m 2 /usr/bin/python3 compute_Cl.py ${dirDR2}/cls_Icut2p0_SN5p0 ${dirDR2}/LoTSS_DR2_v100.srl.fits ${dirDR2}/p_map_Icut2p0_SN5p0.fits.gz ${dirMV}/kmap_rotated.fits ${dirMV}/mask_rot_apo0.200_C1.fits.gz ${dirMV}/nlkk.dat all 2. 5.
    echo "Cls, DR2, p from catalog"
    addqueue -c cls_DR2_pcat -n 1x28 -s -q berg -m 2 /usr/bin/python3 compute_Cl.py ${dirDR2}/cls_Icut2p0_SN5p0_pcat ${dirDR2}/LoTSS_DR2_v100.srl.fits ${dirDR2}/p_map_Icut2p0_SN5p0_fromcat.fits.gz ${dirMV}/kmap_rotated.fits ${dirMV}/mask_rot_apo0.200_C1.fits.gz ${dirMV}/nlkk.dat all 2. 5.
    echo " "
fi
