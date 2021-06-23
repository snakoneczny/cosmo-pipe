import numpy as np
import os
from lotss_corr import Pointings


def dwl_file(url, fname_out, redwl=False, verbose=True, is_LoTSS=True):
    if (not os.path.isfile(fname_out)) or redwl:
        if verbose:
            print(f"Downloading {fname_out} from {url}")
        command = "wget "
        if is_LoTSS:
            command += "--user=$LOTSS_USER --password=$LOTSS_PWD "
        command += f"-O \"{fname_out}\" \"{url}\""
        print(command)
        os.system(command)
    else:
        if verbose:
            print(f"Found {fname_out}")


def download_pointings(pt, which, re_download=False):
    for i, n in enumerate(pt.data['name']):
        n = n.decode()
        url = pt.data[which][i].decode()
        fname_out = pt.prefix_out + f'/{n}_{which}.fits'
        dwl_file(url, fname_out, re_download)


# Download pointing data
pt = Pointings('data/pointings.txt', '/mnt/extraspace/damonge/LensLotss/DR2_data')
download_pointings(pt, 'fr_rms')
pt_dr1 = Pointings('data/pointings_dr1.txt', '/mnt/extraspace/damonge/LensLotss/DR1_data', dr=1)
download_pointings(pt_dr1, 'fr_rms')

# Download source catalog
dwl_file("https://lofar-surveys.org/downloads/DR2/catalogues/LoTSS_DR2_v100.srl.fits",
         "/mnt/extraspace/damonge/LensLotss/DR2_data/LoTSS_DR2_v100.srl.fits")
dwl_file("https://lofar-surveys.org/public/LOFAR_HBA_T1_DR1_catalog_v1.0.srl.fits",
         '/mnt/extraspace/damonge/LensLotss/DR1_data/radio_catalog.fits')
dwl_file("https://lofar-surveys.org/public/LOFAR_HBA_T1_DR1_merge_ID_optical_f_v1.2b_restframe.fits",
         '/mnt/extraspace/damonge/LensLotss/DR1_data/hetdex_optical_ids.fits')
# Download kappa maps (minimum variance)
dwl_file("http://pla.esac.esa.int/pla/aio/product-action?COSMOLOGY.COSMOLOGY_OID=131&COSMOLOGY.FILE_ID=MV.tgz",
         "/mnt/extraspace/damonge/LensLotss/CMBk_MV.tgz", is_LoTSS=False)
dwl_file("http://pla.esac.esa.int/pla/aio/product-action?COSMOLOGY.COSMOLOGY_OID=130&COSMOLOGY.FILE_ID=mask.fits.gz",
         "/mnt/extraspace/damonge/LensLotss/CMBk_mask.fits.gz", is_LoTSS=False)
comm = 'cd /mnt/extraspace/damonge/LensLotss/ ; '
comm += 'tar -xvf CMBk_MV.tgz ; '
comm += 'mv CMBk_mask.fits.gz MV/mask.fits.gz ; '
comm += 'rm CMBk_MV.tgz ; cd $HOME/LensLotss'
os.system(comm)

# Download kappa maps (SZ deprojection)
dwl_file("http://pla.esac.esa.int/pla/aio/product-action?COSMOLOGY.COSMOLOGY_OID=3004&COSMOLOGY.FILE_ID=TT.tgz",
         "/mnt/extraspace/damonge/LensLotss/CMBk_SZdeproj_TT.tgz", is_LoTSS=False)
dwl_file("http://pla.esac.esa.int/pla/aio/product-action?COSMOLOGY.COSMOLOGY_OID=3003&COSMOLOGY.FILE_ID=mask.fits.gz",
         "/mnt/extraspace/damonge/LensLotss/CMBk_SZdeproj_mask.fits.gz", is_LoTSS=False)
comm = 'cd /mnt/extraspace/damonge/LensLotss/ ; '
comm += 'tar -xvf CMBk_SZdeproj_TT.tgz ; '
comm += 'mv TT SZdeproj ; '
comm += 'mv CMBk_SZdeproj_mask.fits.gz SZdeproj/mask.fits.gz ; '
comm += 'rm CMBk_SZdeproj_TT.tgz ; cd $HOME/LensLotss'
os.system(comm)
