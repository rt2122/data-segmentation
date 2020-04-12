def obj_in_pix(nside, ra, dec):
    from math import pi, modf
    sc = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs') 
    theta = sc.galactic.l.degree
    phi = sc.galactic.b.degree
    return hp.ang2pix(nside=2, nest=True, theta=theta, phi=phi, lonlat=True)


