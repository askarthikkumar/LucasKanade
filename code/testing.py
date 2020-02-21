import numpy as np
import cv2
import LucasKanadeAffine as LKA
import scipy.ndimage
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
import InverseCompositionAffine as ICA
import LucasKanade as LK
import LucasKanadeBasis as LKB
import time

theta=5
def sin(angle):
    return np.sin(3.141*angle/180)
def cos(angle):
    return np.cos(3.141*angle/180)

M = np.array([[cos(theta),sin(theta)],[-sin(theta),cos(theta)]])
offset = np.array([-5,10])

aerial = np.load('../data/aerialseq.npy')
It = aerial[:,:,35]
It1 = scipy.ndimage.affine_transform(It,M,offset)
#It1 = aerial[:,:,36]
print("Reference")
print(M)
print(offset)
print("ICA")
t = time.time()
print(ICA.InverseCompositionAffine(It,It1))
print(time.time()-t)
print("LKA")
t = time.time()
print(LKA.LucasKanadeAffine(It,It1))
print(time.time()-t)

cars = np.load('../data/carseq.npy')
rects = np.load('carseqrects.npy')
M = np.array([[1, 0], [0, 1]])
print("Reference")
print(offset)
It = cars[:,:,23]
It1 = scipy.ndimage.affine_transform(It,M,offset)
t = time.time()
print("LK")
print(LK.LucasKanade(It,It1,rects[23]))
print(time.time()-t)

print("Reference")
print(offset)
sylv = np.load('../data/sylvseq.npy')
rects = np.load('sylvseqrects.npy')
bases = np.load('../data/sylvbases.npy')
It = sylv[:,:,45]
It1 = scipy.ndimage.affine_transform(It,M,offset)
t = time.time()
print("LKB")
print(LKB.LucasKanadeBasis(It,It1,rects[45],bases))
print(time.time()-t)
