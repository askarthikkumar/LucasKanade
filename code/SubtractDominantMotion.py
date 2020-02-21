import numpy as np


import cv2
import LucasKanadeAffine as LKA
import InverseCompositionAffine as IKA
import scipy.ndimage
def SubtractDominantMotion(image1, image2):
	# Input:
	#	Images at time t and t+1 
	# Output:
	#	mask: [nxm]
    # put your implementation here
    
    mask = np.ones(image1.shape, dtype=bool)
    #makes timestamping images, keeps notation simple
    It = image1
    It1 = image2
    M = LKA.LucasKanadeAffine(It,It1)

    # M = IKA.InverseCompositionAffine(It,It1)
    #Subtract Dominant Motion
    threshold = 0.1
    It_warped = cv2.warpAffine(It,M,(It.shape[1],It.shape[0]))
    diff = abs(It1-It_warped)
    mask = diff>threshold
    dilated = scipy.ndimage.binary_dilation(mask,structure=np.ones((3,3)))
    # erosion to remove noise
    eroded = scipy.ndimage.binary_erosion(dilated,structure=np.ones((2,2)))
    #to amplify prominent features

    dilated = scipy.ndimage.binary_dilation(eroded,structure=np.ones((3,3)))

    #drive boundaries to zero
    dilated[:10,:] = dilated[:,:10]=dilated[dilated.shape[0]-10:,:]=dilated[:,dilated.shape[1]-10:]=0
    
    assert(dilated.dtype==bool)
    assert(dilated.shape==image1.shape)
    mask = dilated
    return mask
