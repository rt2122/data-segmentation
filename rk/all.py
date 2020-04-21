from only_colab import get_patch
import healpy as hp
import pandas as pd
import numpy as np
from skimage.io import imshow, imsave
from matplotlib import pyplot as plt
from tqdm import tqdm_notebook


def remove_duplicates_patch(patch):
    duplicates = patch[patch.duplicated(subset=["l", "b"], keep='first')]
    coords = set([(l, b) for l, b in zip(duplicates["l"], duplicates["b"])])
    
    params = ['gPSFFlux', 'gKronFlux',  
          'rPSFFlux', 'rKronFlux',  
          'iPSFFlux', 'iKronFlux',  
          'zPSFFlux', 'zKronFlux', 
          'yPSFFlux', 'yKronFlux']
    
    for p in params:
        idx = patch[patch[p+'Err']==-999].index
        patch.loc[idx,p+'Err'] = 999
    
    for l, b in coords:
        index = patch[np.logical_and(patch["l"] == l, patch["b"] == b)].index[0]
        cur_duplicates = duplicates[duplicates["l"] == l][duplicates["b"] == b]
        for p in params:
            err = patch.loc[index, p+'Err']
            min_err = min(cur_duplicates[p+'Err'])
            if err > min_err:
                val = cur_duplicates[cur_duplicates[p+'Err']==min_err][p].values[0]
                patch.loc[index, p] = val
                
    patch.drop_duplicates(subset=["l", "b"], keep='first', inplace=True)
    patch.index = np.arange(patch.shape[0])
    return patch

def calc_pix(patch):
    for i in tqdm_notebook(range(patch.shape[0])):
        theta = patch.iloc[i]["l"]
        phi = patch.iloc[i]["b"]
        patch.loc[i,'pix'] = hp.ang2pix(theta=theta, phi=phi, 
                                         nside=2**17, nest=True, lonlat=True)
        
def pix2dict(matr):
    ans = {}
    for i in range(matr.shape[0]):
        for j in range(matr.shape[1]):
            ans[matr[i, j]] = (i, j)
    return ans


def pic_all_filters(patch, pix_dict):


    '''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
    def pic_one_filter(f, patch, pix_dict, size=4096, base=0):
        result = np.full((size, size), base, dtype=np.float32)
        drop = []
        for i in tqdm_notebook(range(patch.shape[0])):
            pix = patch['pix'].iloc[i]
            flux = patch[f+'KronFlux'].iloc[i]
            if flux == -999:
                flux = patch[f + 'PSFFlux'].iloc[i]
            if flux == -999:
                flux = base
            
            if pix in pix_dict:
                x, y = pix_dict[pix]
            else:
                drop.append(i)
            result[x, y] = flux
        if len(drop) > 0:
            patch.drop(drop, axis='index', inplace=True)
        return result
    '''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''


    ans = []
    for f in ['g', 'r', 'i', 'z', 'y']:
        ans.append(pic_one_filter(f, patch, pix_dict))
    return np.dstack(ans)


def make_mask(ra, dec, dict_pix, cat, cluster_radius=0.08, patch_radius=1.7, nside=2**17, size=4096):

    '''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
    def ra_dec2vec(ra, dec):
        sc = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
        theta = sc.galactic.l.degree
        phi = sc.galactic.b.degree
        vec = hp.ang2vec(lonlat=True, theta = theta, phi = phi)
        return vec


    def find_clusters(cat, ra, dec, patch_radius, cluster_radius, nside=2**17):
        sc = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
        cat_sc = SkyCoord(ra=cat['RAdeg']*u.degree, dec=cat['DEdeg']*u.degree, frame='icrs')
        
        dists = sc.separation(cat_sc).degree
        clusters = cat.iloc[dists < patch_radius]
        
        pixels=[]
        for i in range(clusters.shape[0]):
            vec=ra_dec2vec(clusters['RAdeg'], clusters['DEdeg'])[0]
            pixels.extend(hp.query_disc(nside=nside, vec=vec, radius=np.radians(cluster_radius), nest=True))
        return pixels

    def draw_clusters(dict_pix, pix, size=4096):
        ans=np.zeros((size, size), dtype=np.uint8)
        for p in pix:
            if p in dict_pix:
                ans[dict_pix[p]] = 1
        return ans
    '''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''


    mask = find_clusters(planck, ra, dec, patch_radius, cluster, nside=nside)
    return draw_clusters(dict_pix, mask, size)
