from os.path import join, basename
from astropy.io import fits
import numpy as np
from astropy import wcs
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.wcs.utils import skycoord_to_pixel

class Area:
    var_dict = {'img_conv.fits.gz' : 'img_conv', 
                'exp.fits.gz' : 'exp', 
                'pbg.fits.gz' : 'pbg'}
    
    def __init__(self, data_dir, src_file, layer_name='L3', 
            fldir='allwise_aux/psf_img_allwise_w{}.fits', flnum=1, eps=10, mask_size=(5, 5), cut=1):

        self.number = int(basename(data_dir))
        data_dir = join(data_dir, layer_name)
        self.img_conv = None
        self.exp = None
        self.pbg = None
        self.ir = None
        self.src = None
        self.X = None
        self.Y = None
        self.shape = None
        
        ###'''''''''''''''''''''''''''''''###
        '''extracting data for base layers'''
        ###'''''''''''''''''''''''''''''''###

        for filename in self.var_dict:
            varname = self.var_dict[filename]
            
            arr = fits.getdata(join(data_dir, filename), ext=0)
            np.clip(arr, a_min=0, a_max=None)
            setattr(self, varname, arr)
        
        self.shape = self.img_conv.shape
        

        ###'''''''''''''''''''###
        '''extracting src data'''
        ###'''''''''''''''''''###
        src_table = fits.open(src_file)
        world_img = fits.open(join(data_dir, 'img_conv.fits.gz'))
        world = wcs.WCS(world_img[0].header)
        self.src = []
        for cnt in src_table[1].data:
            self.src.append(list(skycoord_to_pixel(SkyCoord(ra=cnt['RA']*u.degree, dec=cnt['DEC']*u.degree), world)))
        self.src = np.array(self.src, np.int64)



        ###''''''''''''''''''''''''''''''###
        '''make infrared channel and mask'''
        ###''''''''''''''''''''''''''''''###
        def ir(bg, fl, center):
            st = np.array(center) - np.array(fl.shape) // 2
            en = st + np.array(fl.shape)
            fst = np.array([0, 0], np.int64)
            fen = np.array(fl.shape)
            for i, b in enumerate(st < 0):
                if b:
                    fst[i] -= st[i]
                    st[i] = 0
            for i, b in enumerate(en >= np.array(bg.shape)):
                if b:
                    dif = en[i] - bg.shape[i] + 1
                    fen[i] -= dif
                    en[i] -= dif
            bg[st[0]:en[0], st[1]:en[1]] = np.maximum(bg[st[0]:en[0], st[1]:en[1]], fl[fst[0]:fen[0], fst[1]:fen[1]])

            
        self.ir = np.zeros(self.img_conv.shape)
        self.Y = np.zeros(self.img_conv.shape)

        fl = fits.getdata(fldir.format(flnum), ext=0)
        sq = np.ones(mask_size)
        for center in self.src:
            ir(self.ir, fl, center)
            ir(self.Y, sq, center)



        ###''''''''''''''''''###
        '''stack all channels'''
        ###''''''''''''''''''###
        self.X = np.dstack([self.img_conv, self.exp, self.pbg, self.ir])



        return





    def make_pic(self):
        arrs = []
        for var in self.var_dict.values():
            arr = np.copy(getattr(self, var))
            arr /= arr.max()
            arrs.append(arr)
        return np.dstack(arrs).astype(np.uint8)

    def show_ir_mask(self):
        arrs = []
        for var in ['img_conv', 'ir', 'Y']:
            arr = np.copy(getattr(self, var))
            arr /= arr.max()
            arrs.append(arr)
        return np.dstack(arrs).astype(np.float64)
    

    def select(self, nums=None, uint=False):
        var = ['img_conv', 'exp', 'pbg', 'ir']
        if nums is None:
            nums = list(range(len(var)))
        res = [np.copy(getattr(self, var[i])) for i in nums]
        res = np.dstack(res)
        if uint:
            res /= res.max()
            res *= 255
            res = res.astype(np.uint8)
        return res

    def make_cut(self, n_cuts=2):
        var_list = ['img_conv', 'exp', 'pbg', 'ir', 'Y']
        frames = [[] for i in range(cut ** 2)]
        frames_y = []
        for var in var_list:
            arr = np.copy(getattr(self, var))
            hor_split = np.split(arr, cut)
            splits = []
            for stripe in hor_split:
                splits.extend(np.split(stripe, cut, axis=1))

            if var == 'Y':
                frames_y.extend(splits)
            else:
                for i in range(len(frames)):
                    frames[i].append(splits[i])
        for i in range(len(frames)):
            frames[i] = np.dstack(frames[i])

        return frames, frames_y
