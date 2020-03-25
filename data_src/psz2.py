from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt


planck = '/home/rt2122/Desktop/data-segmentation/data_src/Planck_SZ2.fits'


def find_inter_ra_de(a, b, eps):
    num = 0
    for i in range(b.shape[1]):
        ra = b[0, i]
        de = b[1, i]

        if len([x for x in a if abs(x[0] - ra) < eps and abs(x[1] - de) < eps]) > 0:
            num += 1 
    return num

with fits.open(planck) as table:
    table.verify('fix')
    data = table[1].data
    keys = data.names
    redshifts = np.array(data['z'])
    Planck_z = np.count_nonzero(redshifts[redshifts != -1])
    Planck_no_z = np.count_nonzero(redshifts[redshifts == -1])


    planck_names = data['Name']
    planck_names = [name[5:] for name in planck_names]
    planck_names = np.array(planck_names).astype(str)

    radec = [data['radeg'], data['dedeg']]
    radec = np.array(radec).astype(float)
    #print(radec[:, 5])

    mcxc = '/home/rt2122/Desktop/data-segmentation/data_src/MCXC.fits'
    with fits.open(mcxc) as mcxc_table:
        mcxc_table.verify('fix')
        m_data = mcxc_table[1].data
        m_keys = m_data.names
        key_names = m_keys[:3]

        mcxc_names = [m_data[x] for x in key_names]
        mcxc_names = np.array(mcxc_names).astype(str)
        
        mradec = [m_data['radeg'], m_data['dedeg']]
        mradec = np.array(mradec).astype(float)
        #plt.scatter(radec[0], radec[1], c='r')
        #plt.scatter(mradec[0], mradec[1], c='g')
        #plt.show()
        #print(mradec[:, 5])
        MCXCwP = len(m_data)
        print(find_inter_ra_de(radec, mradec, 128.034))
        #print(MCXCwP)
        '''
        for j in range(mcxc_names.shape[1]):
            flag = False
            for n in mcxc_names[:, j]:
                if n in planck_names:
                    flag = True
                    break
            if flag:
                MCXCwP -= 1'''
        #print(MCXCwP)

