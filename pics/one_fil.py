from skimage.io import imsave, imread

s = './check_mask.png'
pix = imread(s)
imsave('./mask.png', pix[:,:,2])
