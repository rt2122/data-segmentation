import numpy as np 
import random as rnd
from matplotlib import pyplot as plt

def gen_gauss_dots(n_dots, xy_range, scale, mu=0, sigma=1):

    ''' xy_range: [[x_st, x_en], [y_st, y_en]] '''
    xy_range = np.array(xy_range).astype(np.int64)

    res = []
    for i in range(n_dots):
        res.append([rnd.gauss(mu, sigma), rnd.gauss(mu, sigma)])
    
    res = np.array(res)
    res *= scale
    res[:, 0] += xy_range[0].mean()
    res[:, 1] += xy_range[1].mean()
    
    #res = [[x, y] for [x, y] in res if (x >= x_range[0]) and (x < x_range[1]) and 
    #                                   (y >= y_range[0]) and (y < y_range[1])]
    
    '''
    for i in range(res.shape[0]):
        for j in range(2):
            if res[i, j] < xy_range[j, 0]:
                dif = int(xy_range[j, 0] - res[i, j])
                dif //= xy_range[j, 1] - xy_range[j, 0]
                res[i, j] += dif * xy_range[j, 1] - xy_range[j, 0]
            if res[i, j] >= xy_range[j, 1]:
                dif = int(res[i, j] - xy_range[j, 1])
                dif //= xy_range[j, 1] - xy_range[j, 0]
                res[i, j] -= dif * xy_range[j, 1] - xy_range[j, 0]
    '''
    for i in range(res.shape[0]):
        for j in range(2):
            if res[i, j] < xy_range[j, 0] or res[i, j] >= xy_range[j, 1]:
                res[i, j] = np.random.randint(xy_range[j, 0], xy_range[j, 1])
    res = np.array(res)

    return res

class Src:
    def __init__(self, x, y, max_rad, max_n, noise=False):
        self.x = x
        self.y = y
        self.rad = np.random.randint(2, max_rad + 1) 
        self.n = np.random.randint(2, max_n + 1)
        if noise:
            self.n = max_n
    def gen_ph(self, label, scale=1, shape=None):
        xy_range = [[self.x - self.rad, self.x + self.rad], 
                [self.y - self.rad, self.y + self.rad]]
        coords = gen_gauss_dots(self.n, xy_range, scale)
        if shape is not None:
            coords = np.array([[x, y] for [x, y] in 
                coords if x >= 0 and y >= 0 and x < shape[0] and y < shape[1]])
        labels = np.array([label] * len(coords)).reshape(len(coords), 1)
        return np.hstack([coords, labels])


def gen_src(n_src, max_rad, max_n, shape):
    coords = gen_gauss_dots(n_src, [[0, shape[0]], [0, shape[1]]], max(shape) * 44) 
    srcs = []
    for [x, y] in coords:
        srcs.append(Src(x, y, max_rad, max_n))
    return np.array(srcs)

def gen_ph_map(srcs, shape, n_noise):
    mtr = np.zeros(shape, dtype=np.uint8)
    all_ph = []
    for i in range(len(srcs)):
        all_ph.extend(srcs[i].gen_ph(i, shape=shape))
    
    noise = Src(shape[0] // 2, shape[1] // 2, max_rad = max(shape) * 4, max_n=n_noise, noise=True)
    all_ph.extend(noise.gen_ph(len(srcs), scale=max(shape) * 440, shape=shape))
    return np.array(all_ph)

def gen_all(n_src, max_rad, max_n, shape, n_noise):
    srcs = gen_src(n_src, max_rad, max_n, shape)
    return gen_ph_map(srcs, shape, n_noise)




alls = gen_all(10, 4, 5, (100, 100), 1000)
plt.figure(num=0, figsize=(10, 10))
plt.scatter(alls[:, 0], alls[:, 1], c=alls[:, 2])
plt.show()

