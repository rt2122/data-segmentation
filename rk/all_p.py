from only_colab import get_patch
import healpy as hp
import pandas as pd
import numpy as np
from skimage.io import imshow, imsave
from matplotlib import pyplot as plt
from tqdm import tqdm_notebook
import re
from os.path import join
from astropy.coordinates import SkyCoord
from astropy import units as u


def remove_duplicates_patch(patch, drop_err=True):
    duplicates = patch.loc[patch.duplicated(subset=["l", "b"], keep='first')]
    coords = set([(l, b) for l, b in zip(duplicates["l"], duplicates["b"])])
    
    params = ['gPSFFlux', 'gKronFlux',  
          'rPSFFlux', 'rKronFlux',  
          'iPSFFlux', 'iKronFlux',  
          'zPSFFlux', 'zKronFlux', 
          'yPSFFlux', 'yKronFlux']
    
    for p in params:
        idx = patch[patch[p+'Err']==-999].index
        patch.loc[idx,p+'Err'] = np.nan
        idx = patch[patch[p]==-999].index
        patch.loc[idx,p]=np.nan

    
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
    if drop_err:
        params = [p + 'Err' for p in params]
        patch.drop(params, axis='columns', inplace=True)

    patch.index = np.arange(patch.shape[0])
    return patch

def calc_pix(patch, nside=2**17):
    patch['pix'] = np.zeros((patch.shape[0]), dtype=np.int64)
    for i in tqdm_notebook(range(patch.shape[0])):
        theta = patch.iloc[i]["l"]
        phi = patch.iloc[i]["b"]
        patch.loc[i,'pix'] = hp.ang2pix(theta=theta, phi=phi, 
                                         nside=nside, nest=True, lonlat=True)
        
def pix2dict(matr):
    ans = {}
    for i in range(matr.shape[0]):
        for j in range(matr.shape[1]):
            ans[matr[i, j]] = (i, j)
    return ans


def pic_all_filters(patch, pix_dict, base=0, size=4096):


    '''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
    def pic_one_filter(f, patch, pix_dict, size=4096, base=0):
        result = np.full((size, size), base, dtype=np.float32)
        drop = []
        for i in tqdm_notebook(range(patch.shape[0])):
            pix = patch['pix'].iloc[i]
            flux = patch[f+'KronFlux'].iloc[i]
            if np.isnan(flux):
                flux = patch[f + 'PSFFlux'].iloc[i]
            if np.isnan(flux):
                flux = base
            
            if pix in pix_dict:
                x, y = pix_dict[pix]
                result[x, y] = flux
            else:
                drop.append(i)
        if len(drop) > 0:
            patch.drop(drop, axis='index', inplace=True)
        return result
    '''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''


    ans = []
    for f in ['g', 'r', 'i', 'z', 'y']:
        ans.append(pic_one_filter(f, patch, pix_dict, size, base))
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


    mask = find_clusters(cat, ra, dec, patch_radius, cluster_radius, nside=nside)
    return draw_clusters(dict_pix, mask, size)


class ClusterFile:

    pnames = pd.DataFrame({'var':     ['typ', 'id_patch', 'ra', 'dec', 'state' , 'num',  'inpix', 'id_list'],
                           'short':   ['t',   'ip',       'ra', 'dec', 's',      'n',    'in',    'il'],
                           'val_type':['s',   'i',        'f',  'f',   's',      'i',    'i',     'i']})

    def __init__(self, name):
        self.params = {'typ' : None,
                'id_patch' : None,
                'ra' : None,
                'dec' : None,
                'state' : None,
                'num' : None,
                'inpix' : None,
                'id_list' : None}
        name = name[:-len('.csv')]
        name = re.split('_', name)
        for n in name:
            for i in range(self.pnames.shape[0]):
                sh_var = self.pnames['short'].iloc[i]
                if n.startswith(sh_var):
                    val = n[len(sh_var):]
                    if self.pnames['val_type'].iloc[i] == 'i':
                        val = int(val)
                    elif self.pnames['val_type'].iloc[i] == 'f':
                        val = float(val)
                    self.params[self.pnames['var'].iloc[i]] = val

    def file(self):
        s = ''
        for p in self.params:
            if not (self.params[p] is None):
                idx = self.pnames[self.pnames['var'] == p].index[0]
                sh = self.pnames['short'].iloc[idx]
                s += sh
                val = self.params[p]
                if re.match('float', str(type(val))):
                    s += '%.4f' % val
                else:
                    s += str(val)
                s += '_'
        s = s[:-1]
        return s + '.csv'




def proc_files(files_list, cdir):
    for f in files_list:
        cf = ClusterFile(f)
        if cf.params['typ'] == 'dat':
            data = pd.read_csv(f)
            first_len = data.shape[0]
            data.index.name='index'
            remove_duplicates_patch(data)
            calc_pix(data)
            print(cf.params['id_patch'], ') Removed:', first_len - data.shape[0])
            cf.params['state'] = 'cl'
            f_clear = cf.file()
            print(f_clear)
            data.to_csv(join(cdir,f_clear))


def make_pic(center_pix, nside=2**11, size=64):
    def get_neighbours(npix, direction=None):
        theta, phi = hp.pix2ang(nside=nside, ipix=npix, nest=True)
        neighbours = hp.get_all_neighbours(nside=nside, theta=theta, phi=phi, nest=True)
        if direction is None:
            return neighbours
        dirs = ['sw', 'w', 'nw', 'n', 'ne', 'e', 'se', 's']
        return neighbours[dirs.index(direction)]
    
    ''' ~~~~~~~~~~> y 
      |  n __nw__ w
      |    |    |
    x | ne |    | sw
      |    |    |
      \/ e ~~se~~ s
         
    '''
    half = size // 2
    ans = np.ones((size, size), dtype=np.int64)
    ans *= -1
    ans[half - 1, half - 1] = center_pix
    for i in range(half - 2, -1, -1):
        ans[i, i] = get_neighbours(ans[i + 1, i + 1], 'n')
    for i in range(1, size):
        ans[i, 0] = get_neighbours(ans[i - 1, 0], 'se')
    for i in tqdm_notebook(range(size)):
        for j in range(1, size):
            if ans[i, j] == -1:
                ans[i, j] = get_neighbours(ans[i, j - 1], 'sw')
    return ans

def ra_dec2n_pix(ra, dec, nside):
    sc = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
    theta = sc.galactic.l.degree
    phi = sc.galactic.b.degree
    npix = hp.ang2pix(nside=nside, nest=True, lonlat=True, theta = theta, phi = phi)
    return npix


