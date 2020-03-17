import numpy as np 
import random as rnd
from matplotlib import pyplot as plt
from skimage.draw import circle
from skimage.io import imshow
from scipy.stats import poisson
from time import time, strftime
import threading


def gen_gauss_dots(n_dots, xy_range, scale, mu=0, sigma=1, re_random=True):

    ''' xy_range: [[x_st, x_en], [y_st, y_en]] '''


    xy_range = np.array(xy_range).astype(np.int64)
    res = []
    for i in range(n_dots):
        res.append([rnd.gauss(mu, sigma), rnd.gauss(mu, sigma)])
    res = np.array(res)


    res *= scale
    res[:, 0] += xy_range[0].mean()
    res[:, 1] += xy_range[1].mean()
    res = res.astype(np.int64)

    if re_random:
        for i in range(res.shape[0]): 
            for j in range(2):
                if not (res[i, j] in range(xy_range[j, 0], xy_range[j, 1], 1)):
                    res[i, j] = np.random.randint(low=xy_range[j, 0], high=xy_range[j, 1], dtype=np.int64)
    else:
        res = [[x, y] for [x, y] in res if x in range(xy_range[0, 0], xy_range[0, 1], 1) and 
                                           y in range(xy_range[1, 0], xy_range[1, 1], 1)]
        res = np.array(res)
    
    return res

class Src: #добавить генерацию протяженного объекта

    scale_mult = 1

    def __init__(self, x, y, max_rad, max_n, noise=False):
        self.x = x
        self.y = y
        #self.rad = np.random.randint(2, max_rad + 1) #Poiss lambda - num ph 
        #self.rad = max_rad
        self.rad = poisson.rvs(mu=max_rad)
        self.n = poisson.rvs(mu=max_n)
        self.ph = None
        self.noise = noise
        if noise:
            self.rad = max_rad
            self.n = max_n

    def gen_ph(self, label, scale=3, shape=None):
        if not self.noise:
            scale *= Src.scale_mult
        xy_range = [[self.x - self.rad, self.x + self.rad + 1], 
                    [self.y - self.rad, self.y + self.rad + 1]]
        if self.noise:
            xy_range = [[0, shape[0]], [0, shape[1]]]
        coords = gen_gauss_dots(self.n, xy_range, scale, re_random=self.noise, sigma=self.rad // 20)
        if shape is not None:
            coords = np.array([[x, y] for [x, y] in coords 
                if x in range(shape[0]) and y in range(shape[1])])
        if coords.shape[0] == 0:
            return None

        labels = np.array([label] * len(coords)).reshape(len(coords), 1)
        self.ph = np.hstack([coords, labels])


def gen_src(n_src, max_rad, max_n, shape):
    coords = gen_gauss_dots(n_dots=n_src, xy_range=[[0, shape[0]], [0, shape[1]]], 
            scale=max(shape) * 4) 
    srcs = []
    n_ph = 0
    for [x, y] in coords:
        srcs.append(Src(x, y, max_rad, max_n))
        n_ph += srcs[-1].n
    return np.array(srcs), n_ph

def gen_ph_map(srcs, shape, n_noise, n_ph):

    class PhotonThread(threading.Thread):
        def __init__(self, i, srcs, shape, scale=1):
            threading.Thread.__init__(self)
            self.i = i
            self.srcs = srcs
            self.shape = shape
            self.scale = scale

        def run(self):
            self.srcs[self.i].gen_ph(self.i, shape=self.shape, scale=self.scale)

    for i in range(len(srcs)):
        t = PhotonThread(i, srcs, shape)
        t.start()

    all_ph = []
    for i in range(len(srcs)):
        if not (srcs[i].ph is None):
            all_ph.append(srcs[i].ph)

    noise = Src(shape[0] // 2, shape[1] // 2, max_rad = max(shape) // 2, max_n=n_noise, noise=True) 
    noise.gen_ph(len(srcs), scale=max(shape) * 440, shape=shape)
    all_ph.append(noise.ph)
    all_ph = np.vstack(all_ph)

    return all_ph

def gen_all(n_src, max_rad, max_n, shape, n_noise):
    srcs, n_ph = gen_src(n_src, max_rad, max_n, shape)
    ph_map = gen_ph_map(srcs, shape, n_noise, n_ph) 
    return ph_map, srcs

def gen_train(n_src, max_rad, max_n, shape, d_noise, n_out=None, scale_mult=1, steps=1):
    Src.scale_mult = scale_mult
    while 4:
        X = []
        Y = []
        for j in range(steps):
            if n_out is None:
                n_out = n_src
            n_noise = int(d_noise * shape[0] * shape[1])
            phs, srcs = gen_all(n_src, max_rad, max_n, shape, n_noise)
            x = np.zeros(shape, np.uint8)
            y = np.zeros(list(shape)[:-1] + [n_out], np.uint8)
            phs = phs.astype(np.int64)
            for ph in phs:
                x[ph[0], ph[1], 0] = 1
            for i in range(n_src):
                c = circle(srcs[i].x, srcs[i].y, srcs[i].rad, list(shape)[:-1])
                y[:, :, i][c] = 1
            X.append(x)
            Y.append(y)
        yield np.array(X), np.array(Y)

def random_colour_circles(Y):

    y_pic = np.zeros(list(Y.shape)[:-1] + [3], np.int32)

    for i in range(Y.shape[-1]):
        R = rnd.randint(0, 255)
        G = rnd.randint(0, 255)
        B = rnd.randint(0, 255)

        y_pic[0, :, :, 0] += (R * Y[0, :, :, i]).astype(np.int32)
        y_pic[0, :, :, 1] += (G * Y[0, :, :, i]).astype(np.int32)
        y_pic[0, :, :, 2] += (B * Y[0, :, :, i]).astype(np.int32)
    np.clip(y_pic, 0, 255, y_pic)
    return y_pic[0]

def show_x_y(X, Y, ans=None, num=0):
    if ans is None:
        plt.figure(num=num, figsize=(12, 6))
        fig, axes = plt.subplots(1, 2, num=0)
        axes[0].imshow(X[0, :, :, 0] * 255)
        axes[0].set_title("Карта расположения фотонов")
        axes[1].imshow(random_colour_circles(Y))
        axes[1].set_title("Источники")
    else:
        plt.figure(num=num, figsize=(12, 6))
        fig, axes = plt.subplots(1, 3, num=0)
        axes[0].imshow(X[0, :, :, 0] * 255)
        axes[0].set_title("Карта расположения фотонов")
        axes[1].imshow(random_colour_circles(Y))
        axes[1].set_title("Источники")
        axes[2].imshow(random_colour_circles(ans))
        axes[2].set_titile("Результат работы нейросети")

    plt.show()



rnd.seed(0)

for X, Y in gen_train(12, 50, 120, (512, 512, 1), 0.02, scale_mult=4):
    show_x_y(X, Y)
    break
