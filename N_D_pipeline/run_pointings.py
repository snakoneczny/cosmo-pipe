from utils import Pointings
import os


def find_unique_batch():
    found = False
    ifile = 0
    while not found:
        fname = f'batch{ifile}.txt'
        if not os.path.isfile(fname):
            found = True
        ifile += 1
    return fname

pt_dr2 = Pointings('data/pointings.txt', '/mnt/extraspace/damonge/LensLotss/DR2_data/')
pt_dr1 = Pointings('data/pointings_dr1.txt', '/mnt/extraspace/damonge/LensLotss/DR1_data/', dr=1)
for pt in [pt_dr1, pt_dr2]:
    n_pointings = len(pt.data['name'])
    n_added = 0
    for i_n, n in enumerate(pt.data['name']):
        n = n.decode()
        #fname = pt.prefix_out + f'/{n}_fr_res.fits'
        fname = pt.prefix_out + f'/{n}_fr_rms.fits'
        if not os.path.isfile(fname):
            print(f'{n} Missing {fname}')
            exit(1)
        #fname_o = pt.prefix_out + f'/map_{n}.fits.gz'
        fname_o = pt.prefix_out + f'/map_rms_{n}.fits.gz'
        if os.path.isfile(fname_o):
            print(f'{n} skipped')
            continue

        print(f"adding {i_n}-{n}")
        if n_added == 0:
            i_n0 = i_n
            stout = ""
        stout += f"{fname} {fname_o}\n"
        n_added += 1
        print(i_n, n_pointings)
        if (n_added == 10) or (i_n == n_pointings-1):
            fname_batch = find_unique_batch()
            f = open(fname_batch, "w")
            f.write(stout)
            f.close()
            comment = f'{i_n0}_{i_n}_{fname_batch}'
            queue = 'cmb'
            command = f'addqueue -c {comment} -n 1x1 -q {queue} -m 24 /usr/bin/python3 '
            #command = 'python3 '
            command += f'make_pointing_map.py {fname_batch}'
            print(command)
            os.system(command)
            n_added = 0
