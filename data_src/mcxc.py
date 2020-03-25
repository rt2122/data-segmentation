from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt


filename = '/home/rt2122/Desktop/data-segmentation/data_src/MCXC.fits'
with fits.open(filename) as table:
    table.verify('fix')
    data = table[1].data
    keys = data.names
    print(keys)
    print(len(data))
    '''
    redshifts = np.array(data['z'])
    Planck_z = np.count_nonzero(redshifts[redshifts != -1]))
    Planck_no_z = np.count_nonzero(redshifts[redshifts == -1]))
    '''

