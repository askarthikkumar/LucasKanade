import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanade(It, It1, rect, p0 = np.zeros(2)):
	# Input: 
	#   It: template image           (full image or exactly the cropped template image?)
	#	It1: Current image
	#	rect: Current position of the car
	#	(top left, bot right coordinates)
	#	p0: Initial movement vector [dp_x0, dp_y0]
	# Output:
	#	p: movement vector [dp_x, dp_y]
	
    # Put your implementation here
    
    #swap rect coords to row and col
    rect = np.copy(rect)
    rect[1],rect[0] = rect[0],rect[1]
    rect[2],rect[3] = rect[3],rect[2]
    p = np.copy(np.array([p0[1],p0[0]]))
    row = np.arange(0,It.shape[0])
    col = np.arange(0,It.shape[1])
    It_cont = RectBivariateSpline(row,col,It)
    row = np.arange(0,It1.shape[0])
    col = np.arange(0,It1.shape[1])
    It1_cont = RectBivariateSpline(row,col,It1)

    #scipy.interpolate uses array-like indexing so dx is actually dy in cartesian
    row_coord,col_coord = np.meshgrid(row,col)


    # It1_xgrad_cont = RectBivariateSpline(x,y,It1_xgrad)
    # It1_ygrad_cont = RectBivariateSpline(x,y,It1_ygrad)
    #It1_grad = np.stack((It1_xgrad,It1_ygrad),axis=2)
    delta_p=np.ones((1,2))

    #let x in template be a set of coords given as two arrays which we can update with p
    temp_length = round(rect[2]-rect[0]+1)
    temp_width = round(rect[3]-rect[1]+1)
    top,left = rect[0],rect[1]
    row_temp_coords,col_temp_coords = np.meshgrid(np.arange(0,temp_length),np.arange(0,temp_width))
    
    #translate them to x1_temp and y1_temp
    row_temp_coords+=top
    col_temp_coords+=left
#     print(x_temp_coords[0])
    while np.linalg.norm(delta_p.T[0])>1e-2:

    #1.calculate A

    #It1' (in x and y) and Warp with current p
        warped_row_coords = row_temp_coords+p[0]
        warped_col_coords = col_temp_coords+p[1]

        It1_row_grad_warped = It1_cont.ev(warped_row_coords,warped_col_coords,dx=1,dy=0).T
        It1_col_grad_warped = It1_cont.ev(warped_row_coords,warped_col_coords,dx=0,dy=1).T

        It1_grad_warped = np.stack((It1_row_grad_warped,It1_col_grad_warped),axis=2)

    #Reshape It1' from m*n*2 to (mn)*2
        m,n,_ = It1_grad_warped.shape
        It1_grad_warped = np.reshape(It1_grad_warped, (m*n,2))

    #Multiply It1'_warped with W'
        warp_grad = np.array([[1,0],[0,1]])
        A = It1_grad_warped@warp_grad

    #2.calculate b
    #subtract It-It1_warped
        b = It_cont.ev(row_temp_coords,col_temp_coords).T-It1_cont.ev(warped_row_coords,warped_col_coords).T

    #reshape b to the same shape as A (m*n,2)
        b = np.reshape(b, (m*n,1))

    #3.perform lstsq to get delta_p and update delta_p
        delta_p,_,rank,_ = np.linalg.lstsq(A,b,rcond=None)

    #4.update p
        p = p + delta_p.T[0]
    #     print(delta_p)
        # print(p)

    #bring p to x and y coords
    return np.array([p[1],p[0]])