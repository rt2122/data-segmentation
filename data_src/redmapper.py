from astropy.io import fits
import numpy as np

filename = '/home/rt2122/Desktop/data-segmentation/data_src/redmapper_dr8_public_v6.3_catalog.fits'

with fits.open(filename) as table:
    table.verify('fix')
    print(table.info())
    data = table[1].data
    keys = data.names
    print(keys)
    print(np.count_nonzero(data['LAMBDA'] > 50))
    print(np.count_nonzero(data['LAMBDA'] > 30))
