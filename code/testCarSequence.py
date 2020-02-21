import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
import LucasKanade as LK

# write your script here, we recommend the above libraries for making your animation
carseq = np.load('../data/carseq.npy')

i=0
rect_list = []
rect = np.array([59,116,145,151],dtype='float')
rect_list.append(np.copy(rect))
width = rect_list[0][2]-rect_list[0][0]
length = rect_list[0][3]-rect_list[0][1]
for i in range(0,carseq.shape[2]-1):
    im1 = carseq[:,:,i]
    im2 = carseq[:,:,i+1]

    p = LK.LucasKanade(im1,im2,np.copy(rect))

    rect[0]+=p[0]
    rect[1]+=p[1]
    rect[2] = rect[0]+width
    rect[3] = rect[1]+length
    # print(i)
    # print(p)
    # print(rect)
    rect_list.append(np.copy(rect))
    
    # i+=1

    # uncomment for visualisation
    '''
    frame = i+1
    fig,ax = plt.subplots(1)
    ax.imshow(carseq[:,:,frame],cmap='gray')
    
    #matplotlib.patches requires (left,top) coords
    patch = patches.Rectangle((rect_list[frame][0],rect_list[frame][1]),width,length,edgecolor='r',facecolor='none')
    ax.add_patch(patch)
    
    plt.show(block=False)
    plt.pause(0.01)
    plt.close()
'''
# print(len(rect_list))

# uncomment for saving files
'''
frames_req = [1,100,200,300,400]  #frame-1 or frame?????

for frame in frames_req:
    fig,ax = plt.subplots(1)
    ax.imshow(carseq[:,:,frame],cmap='gray')
    length = rect_list[frame][3]-rect_list[frame][1]
    width = rect_list[frame][2]-rect_list[frame][0]
    #matplotlib.patches requires (left,top) coords
    patch = patches.Rectangle((rect_list[frame][0],rect_list[frame][1]),width,length,edgecolor='r',facecolor='none')
    ax.add_patch(patch)
    plt.savefig(str(frame)+'.png')
'''
carseqrects = np.array(rect_list)


assert(carseqrects.shape == (carseq.shape[2],4))
np.save('carseqrects.npy',np.array(rect_list))
