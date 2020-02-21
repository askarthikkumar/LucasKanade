import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches

# write your script here, we recommend the above libraries for making your animation
import LucasKanade as LK
import LucasKanadeBasis as LKB

# import importlib
# importlib.reload(LKB)
frames = np.load('../data/sylvseq.npy')
bases = np.load('../data/sylvbases.npy')
frames_req = [1, 200, 300, 350 ,400]
i=0
rect_list = []
rect = np.array([101,61,155,107],dtype='float')

#for comparison with LK
rect1 = np.copy(rect)

rect_list.append(np.copy(rect))
width = rect_list[0][2]-rect_list[0][0]
length = rect_list[0][3]-rect_list[0][1]

for i in range(0,frames.shape[2]-1):
    im1 = frames[:,:,i]
    im2 = frames[:,:,i+1]
    # print('lengths')
    # print(length)
    # print(width)
    p = LKB.LucasKanadeBasis(im1,im2,np.copy(rect),bases)
    rect[0]+=p[0]
    rect[1]+=p[1]
    rect[2] = rect[0]+width
    rect[3] = rect[1]+length
    

    rect_list.append(np.copy(rect))

# uncomment for visualisation and saving
'''
    frame = i+1
    fig,ax = plt.subplots(1)
    ax.imshow(frames[:,:,frame],cmap='gray')

    #to compare with LK
    p1 = LK.LucasKanade(im1,im2,np.copy(rect1))
    rect1[0]+=p1[0]
    rect1[1]+=p1[1]
    rect1[2] = rect1[0]+length
    rect1[3] = rect1[1]+width
    patch1 = patches.Rectangle((rect1[0],rect1[1]),width,length,edgecolor='g',facecolor='none')
    ax.add_patch(patch1)

    #matplotlib.patches requires (left,top) coords
    patch = patches.Rectangle((rect_list[frame][0],rect_list[frame][1]),width,length,edgecolor='r',facecolor='none')
    ax.add_patch(patch)
    plt.legend((patch, patch1), ('With Basis', 'Without Basis'))
    
    if frame in frames_req:
        plt.savefig(str(frame)+'_basis.png')
    plt.show(block=False)
    plt.pause(0.01)
    plt.close()

  '''

framesrects = np.array(rect_list)

assert(framesrects.shape == (frames.shape[2],4))
np.save('sylvseqrects.npy',np.array(rect_list))