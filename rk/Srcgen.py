import numpy as np 
import random as rnd
from matplotlib import pyplot as plt
from skimage.draw import circle
from skimage.io import imshow

def gen_gauss_dots(n_dots, xy_range, scale, mu=0, sigma=1, re_random=True):

    ''' xy_range: [[x_st, x_en], [y_st, y_en]] '''
    xy_range = np.array(xy_range).astype(np.int64)

    res = []
    for i in range(n_dots):
        res.append([rnd.gauss(mu, sigma), rnd.gauss(mu, sigma)])
    
    res = np.array(res).astype(np.int64)
    res *= scale
    res[:, 0] += xy_range[0].mean()
    res[:, 1] += xy_range[1].mean()

    if re_random:
        for i in range(res.shape[0]): 
            for j in range(2):
                if res[i, j] in range(xy_range[j, 0], xy_range[j, 1]):
                    res[i, j] = np.random.randint(xy_range[j, 0], xy_range[j, 1])
    else:
        res = [[x, y] for [x, y] in res if x in range(xy_range[0, 0], xy_range[0, 1]) and 
                                           y in range(xy_range[1, 0], xy_range[1, 1])]
        res = np.array(res)

    return res

class Src: #добавить генерацию протяженного объекта

    def __init__(self, x, y, max_rad, max_n, noise=False):
        self.x = x
        self.y = y
        self.rad = np.random.randint(2, max_rad + 1) #Poiss lambda - num ph 
        self.n = np.random.randint(2, max_n + 1)
        if noise:
            self.rad = max_rad
            self.n = max_n

    def gen_ph(self, label, scale=1, shape=None):
        xy_range = [[self.x - self.rad, self.x + self.rad + 1], 
                    [self.y - self.rad, self.y + self.rad + 1]]
        coords = gen_gauss_dots(self.n, xy_range, scale)
        if shape is not None:
            coords = np.array([[x, y] for [x, y] in coords 
                if x in range(shape[0]) and y in range(shape[1])])

        labels = np.array([label] * len(coords)).reshape(len(coords), 1)
        return np.hstack([coords, labels])


def gen_src(n_src, max_rad, max_n, shape):
    coords = gen_gauss_dots(n_src, [[0, shape[0]], [0, shape[1]]], max(shape) * 44) 
    srcs = []
    for [x, y] in coords:
        srcs.append(Src(x, y, max_rad, max_n))
    return np.array(srcs)

def gen_ph_map(srcs, shape, n_noise):
    all_ph = None 
    for i in range(len(srcs)):
        ph = srcs[i].gen_ph(i, shape=shape)
        if all_ph is None:
            all_ph = ph
        else:
           all_ph = np.vstack([all_ph, ph])
    
    noise = Src(shape[0] // 2, shape[1] // 2, max_rad = max(shape) * 4, max_n=n_noise, noise=True) 
    all_ph = np.vstack([all_ph, noise.gen_ph(len(srcs), scale=max(shape) * 440, shape=shape)])
    return all_ph

def gen_all(n_src, max_rad, max_n, shape, n_noise):
    srcs = gen_src(n_src, max_rad, max_n, shape)
    return gen_ph_map(srcs, shape, n_noise), srcs

def gen_train(n_src, max_rad, max_n, shape, d_noise):
    while 4:
        n_noise = int(d_noise * shape[0] * shape[1])
        phs, srcs = gen_all(n_src, max_rad, max_n, shape, n_noise)
        X = np.zeros(shape, np.uint8)
        Y = np.zeros(shape, np.uint8)
        phs = phs.astype(np.int64)
        for ph in phs:
            X[ph[0], ph[1], 0] = 1
        for src in srcs:
            c = circle(src.x, src.y, src.rad, shape)
            Y[c] = 1
        yield np.array([X]), np.array([Y])

rnd.seed(0)

for X, Y in gen_train(50, 10, 20, (500, 500), 0.1):
    plt.figure(num=0)
    imshow(X * 255)
    plt.figure(num=1)
    imshow(Y * 255)
    plt.show()
    break
