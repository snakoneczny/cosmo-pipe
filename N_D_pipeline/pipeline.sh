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
run_DR1_xis=true
run_DR2_xis=true


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

# DR1 power spectra
if [ $run_DR1_cls = true ] ; then
    echo "Cls, DR1, all pointings, radio catalog"
    addqueue -c cls_DR1_all_radiocat -n 1x24 -s -q cmb -m 2 /usr/bin/python3 -m lotss_corr.clcalc params/params_dr1_all_radiocat.yml
    echo "Cls, DR1, good pointings, radio catalog"
    addqueue -c cls_DR1_good_radiocat -n 1x24 -s -q cmb -m 2 /usr/bin/python3 -m lotss_corr.clcalc params/params_dr1_good_radiocat.yml
    echo "Cls, DR1, all pointings, VAC"
    addqueue -c cls_DR1_all_vac -n 1x24 -s -q cmb -m 2 /usr/bin/python3 -m lotss_corr.clcalc params/params_dr1_all_VAC.yml
    echo "Cls, DR1, good pointings, VAC"
    addqueue -c cls_DR1_good_vac -n 1x24 -s -q cmb -m 2 /usr/bin/python3 -m lotss_corr.clcalc params/params_dr1_good_VAC.yml
    echo "Cls, DR1, all pointings, radio catalog, p from cat"
    addqueue -c cls_DR1_all_radiocat_pcat -n 1x24 -s -q cmb -m 2 /usr/bin/python3 -m lotss_corr.clcalc params/params_dr1_all_radiocat_pcat.yml
    echo " "
fi

# DR2 power spectra
if [ $run_DR2_cls = true ] ; then
    echo "Cls, DR2"
    addqueue -c cls_DR2 -n 1x24 -s -q cmb -m 2 /usr/bin/python3 -m lotss_corr.clcalc params/params_dr2.yml
    echo "Cls, DR2, p from catalog"
    addqueue -c cls_DR2_pcat -n 1x24 -s -q cmb -m 2 /usr/bin/python3 -m lotss_corr.clcalc params/params_dr2_pcat.yml
    echo " "
fi

# DR1 power spectra
if [ $run_DR1_xis = true ] ; then
    echo "Xis, DR1, all pointings, radio catalog"
    addqueue -c xis_DR1_all_radiocat -n 1x24 -s -q cmb -m 2 /usr/bin/python3 -m lotss_corr.xicalc params/params_dr1_all_radiocat.yml
    echo "Xis, DR1, good pointings, radio catalog"
    addqueue -c xis_DR1_good_radiocat -n 1x24 -s -q cmb -m 2 /usr/bin/python3 -m lotss_corr.xicalc params/params_dr1_good_radiocat.yml
    echo "Xis, DR1, all pointings, VAC"
    addqueue -c xis_DR1_all_vac -n 1x24 -s -q cmb -m 2 /usr/bin/python3 -m lotss_corr.xicalc params/params_dr1_all_VAC.yml
    echo "Xis, DR1, good pointings, VAC"
    addqueue -c xis_DR1_good_vac -n 1x24 -s -q cmb -m 2 /usr/bin/python3 -m lotss_corr.xicalc params/params_dr1_good_VAC.yml
    echo "Xis, DR1, all pointings, radio catalog, p from cat"
    addqueue -c xis_DR1_all_radiocat_pcat -n 1x24 -s -q cmb -m 2 /usr/bin/python3 -m lotss_corr.xicalc params/params_dr1_all_radiocat_pcat.yml
    echo " "
fi

# DR2 power spectra
if [ $run_DR2_xis = true ] ; then
    echo "Xis, DR2"
    addqueue -c xis_DR2 -n 1x24 -s -q cmb -m 2 /usr/bin/python3 -m lotss_corr.xicalc params/params_dr2.yml
    echo "Xis, DR2, p from catalog"
    addqueue -c xis_DR2_pcat -n 1x24 -s -q cmb -m 2 /usr/bin/python3 -m lotss_corr.xicalc params/params_dr2_pcat.yml
    echo " "
fi
