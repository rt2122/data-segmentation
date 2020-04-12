def n_src_in_radius(cat, center, radius):
    #center : [ra dec]
    dists = ra_dec_distance(center[0], center[1], cat['RAdeg'], cat['DEdeg'])
    return np.count_nonzero(dists < radius.degree) 

def n_pix2ra_dec(npix, nside):
    theta, phi = hp.pix2ang(nside=nside, ipix=[npix], nest=True, lonlat=True)
    sc = SkyCoord(l=theta*u.degree, b=phi*u.degree, frame='galactic')
    return sc.icrs.ra.degree, sc.icrs.dec.degree

def gen_centers(cat, n, radius=astropy.coordinates.Angle('1d'), nside=2**11, func=None):
    npix = hp.nside2npix(nside)
    a = np.arange(npix)
    if not (func is None):
        a = a[func(a, nside=nside)]
    ans = []
    while len(ans) < n:
        num = np.random.choice(a)
        if n_src_in_radius(cat, n_pix2ra_dec(num, nside), radius) > 0:
            ans.append(num)
    return np.array(ans)

def in_nth_pix(n, nside, nbig=6, nbigside=2):
    vec = hp.pix2vec(nest=True, nside=nside, ipix=n)
    return hp.vec2pix(nside=nbigside, x=vec[0], y=vec[1], z=vec[2], nest=True) == nbig
