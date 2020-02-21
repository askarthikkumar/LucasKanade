import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches

# write your script here, we recommend the above libraries for making your animation
import SubtractDominantMotion as SDM

frame_req = [30, 60, 90, 120]
aerial = np.load('../data/aerialseq.npy')
masks = []
for i in range(0,aerial.shape[2]-1):

    # comment the if-block for visualisation
    if (i+1) not in frame_req:
        continue

    It = aerial[:,:,i]
    It1 = aerial[:,:,i+1]
    mask = SDM.SubtractDominantMotion(It,It1)
    if (i+1) in frame_req:
        masks.append(np.copy(mask));

    
    # uncommment for visualisation
    '''
    fig = plt.figure()
    plt.imshow(It1,cmap='gray')
    plt.imshow(mask,alpha=0.2,cmap='viridis')
    if i+1 in frame_req:
        plt.savefig(str(i+1)+'_aerial.png')
    plt.show(block=False)
    plt.pause(0.01)
    plt.close()
    '''
masks = np.dstack(masks)
assert(masks.shape==(aerial.shape[0],aerial.shape[1],4))
np.save('aerialseqrects.npy',masks)
    
