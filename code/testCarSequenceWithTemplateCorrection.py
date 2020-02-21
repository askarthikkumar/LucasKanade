import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches

# write your script here, we recommend the above libraries for making your animation

import LucasKanade as LK

carseq = np.load('../data/carseq.npy')
carseqrects = np.load('carseqrects.npy')
T1 = carseq[:,:,0]
rect_list = []
rect = np.array([59,116,145,151],dtype='float')
rect_list.append(np.copy(rect))
length = rect_list[0][3]-rect_list[0][1]
width = rect_list[0][2]-rect_list[0][0]
p_absolut = np.zeros(2)
rect_1 = np.copy(rect_list[0])

for i in range(0,carseq.shape[2]-1):
    # print('-------------------------------------')
    # print(i)
    im1 = carseq[:,:,i]
    im2 = carseq[:,:,i+1]

    p_rel = LK.LucasKanade(im1,im2,np.copy(rect))
    # print('p_rel')
    # print(p_rel)
    # print('rect')
    # print(rect)

    # rect[0]+=p_rel[0]
    # rect[1]+=p_rel[1]
    # rect[2]+=p_rel[0]
    # rect[3]+=p_rel[1]

    #we need p wrt to rect of carseq frame 0
    p_absolut[0],p_absolut[1] = (rect[0]+p_rel[0]-rect_1[0]),(rect[1]+p_rel[1]-rect_1[1]) 
    # print('p_absolut')
    # print(p_absolut)
    p_updated = LK.LucasKanade(T1,im2,np.copy(rect_list[0]),np.copy(p_absolut))
    # print('rect_list[0]')
    # print(rect_list[0])
    # print('p_star')
    # print(p_updated)

    # print('Normal Lucas Kanade')
    # print(carseqrects[i+1])

    # always update with p*

    rect[0]=(p_updated[0]+rect_1[0])
    rect[1]=(p_updated[1]+rect_1[1])
    rect[2]=(width+rect[0])
    rect[3]=(length+rect[1])
    '''
    if np.linalg.norm(p_absolut-p_updated)<3000:
        print('update')
        # print(p_updated)
        # print(rect_1)
        # p_absolut = p_updated
        # print(i)
        # print('update')
        # print(p_updated)
        # print(p_rel)``
        # print(p_absolut)
        
        rect[0]=(p_updated[0]+rect_1[0])
        rect[1]=(p_updated[1]+rect_1[1])
        rect[2]=(width+rect[0])
        rect[3]=(length+rect[1])
        print(rect)
    else:
        # print(i)
        print('update!')
        # print(p_updated)
        # print(p_rel)
        # print(p_absolut)
        rect[0]+=p_rel[0]
        rect[1]+=p_rel[1]
        rect[2] = rect[0]+width
        rect[3] = rect[1]+length
    # print(rect)
    '''
    rect_list.append(np.copy(rect))

    # uncomment for visualisation
    '''
    frame = i+1
    fig,ax = plt.subplots(1)
    ax.imshow(carseq[:,:,frame],cmap='gray')
    
    #matplotlib.patches requires (left,top) coords
    patch = patches.Rectangle((rect_list[frame][0],rect_list[frame][1]),width,length,edgecolor='r',facecolor='none')
    ax.add_patch(patch)
    patch1 = patches.Rectangle((carseqrects[frame][0],carseqrects[frame][1]),width,length,edgecolor='g',facecolor='none')
    ax.add_patch(patch1)
    plt.legend((patch, patch1), ('With Template Correction', 'Without Template Correction'))
    
    plt.show(block=False)
    plt.pause(0.01)
    plt.close()
    
    '''


# uncomment for saving images
'''
frames_req = [1,100,200,300,400]  #frame-1 or frame?????

for frame in frames_req:
    fig,ax = plt.subplots(1)
    ax.imshow(carseq[:,:,frame],cmap='gray')

    #matplotlib.patches requires (left,top) coords
    patch = patches.Rectangle((rect_list[frame][0],rect_list[frame][1]),width,length,edgecolor='r',facecolor='none', label='With Template Correction')
    ax.add_patch(patch)
    patch1 = patches.Rectangle((carseqrects[frame][0],carseqrects[frame][1]),width,length,edgecolor='g',facecolor='none', label='Without Template Correction')
    ax.add_patch(patch1)
    plt.legend((patch, patch1), ('With Template Correction', 'Without Template Correction'))
    plt.savefig(str(frame)+'_wcrt.png')
'''

carseqrects = np.array(rect_list)

assert(carseqrects.shape == (carseq.shape[2],4))
np.save('carseqrects-wcrt.npy',np.array(rect_list))
