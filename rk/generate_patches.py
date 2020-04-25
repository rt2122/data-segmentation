import astropy
import numpy as np
import healpy as hp
from astropy.coordinates import SkyCoord
from astropy import units as u


def ra_dec_distance(ra, dec, ra1, dec1):
    import numpy as np
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    c1 = SkyCoord(ra=ra, dec=dec, unit='deg')
    c2 = SkyCoord(ra=ra1, dec=dec1, unit='deg')
    sep = c1.separation(c2)
    return sep.degree

def n_src_in_radius(cat, center, radius):
    #center : [ra dec]
    dists = ra_dec_distance(center[0], center[1], cat['RAdeg'], cat['DEdeg'])
    return np.count_nonzero(dists < radius) 

def n_pix2ra_dec(npix, nside):
    theta, phi = hp.pix2ang(nside=nside, ipix=[npix], nest=True, lonlat=True)
    sc = SkyCoord(l=theta*u.degree, b=phi*u.degree, frame='galactic')
    return sc.icrs.ra.degree, sc.icrs.dec.degree

def gen_centers(cat, n, radius, nside=2**11, func=None, part=512, n_src=1):
    npix = hp.nside2npix(nside)
    a = None
    if nside > 2**11:
        a = np.arange(npix // part)
        a *= part
    else:
        a = np.arange(npix)
    print(a.shape)
    if not (func is None):
        a = a[func(a, nside=nside)]
    ans = []
    while len(ans) < n:
        num = np.random.choice(a)
        if n_src_in_radius(cat, n_pix2ra_dec(num, nside), radius) >= n_src:
            ans.append(num)
        else:
            np.delete(a, np.argwhere(a == num))
    return np.array(ans)

def in_nth_pix(n, nside, nbig=6, nbigside=2):
    vec = hp.pix2vec(nest=True, nside=nside, ipix=n)
    return hp.vec2pix(nside=nbigside, x=vec[0], y=vec[1], z=vec[2], nest=True) == nbig
