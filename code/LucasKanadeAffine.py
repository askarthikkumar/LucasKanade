import numpy as np
from scipy.interpolate import RectBivariateSpline

import scipy.ndimage

def get_mask(row, col, m , n):
	row_positive = row>=0
	col_positive = col>=0
	row_upper = row<m
	col_upper = col<n
	row_mask = np.logical_and(row_positive,row_upper)
	col_mask = np.logical_and(col_positive,col_upper)
	return np.logical_and(row_mask,col_mask)

def LucasKanadeAffine(It, It1):
	# Input: 
	#	It: template image
	#	It1: Current image
	# Output:
	#	M: the Affine warp matrix [2x3 numpy array]
    # put your implementation here
	# Input: 
#	It: template image
#	It1: Current image
# Output:
#	M: the Affine warp matrix [2x3 numpy array]
# put your implementation here
	M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

	row = np.arange(0,It.shape[0])
	col = np.arange(0,It.shape[1])

	It_cont = RectBivariateSpline(row,col,It)
	It1_cont = RectBivariateSpline(row,col,It1)

	#scipy.interpolate uses array-like indexing so dx is actually dy in cartesian
	row_coords,col_coords = np.meshgrid(row,col)
	row_coords = row_coords.flatten()
	col_coords = col_coords.flatten()

	m,n = np.shape(It)

	delta_p=np.ones((1,6))
	p = np.zeros(6)

	i=0
	while np.linalg.norm(delta_p.flatten())>1e-2:
		i+=1
		coords = M[:,:2]@np.stack((col_coords,row_coords),axis=0)+np.reshape(M[:,2],(2,1))
		new_col_coords = coords[0]
		new_row_coords = coords[1]
		# mask = np.logical_and(np.logical_and(new_col_coords>=0,new_row_coords>=0),np.logical_and(new_col_coords<n,new_row_coords<m))
		mask = get_mask(new_row_coords,new_col_coords,m,n)
		# print('mask')
		# print(len(mask))
		new_col = new_col_coords[mask]
		new_row = new_row_coords[mask]
		col = col_coords[mask]
		row = row_coords[mask]


		#1.calculate A

		#It1' (in x and y) and Warp with current p

		It1_col_grad = It1_cont.ev(new_row,new_col,dx=0,dy=1)
		It1_row_grad = It1_cont.ev(new_row,new_col,dx=1,dy=0)

		#Multiply It1'_warped with W'
		#warp jacobian is [[x y 1 0 0 0],[0 0 0 x y 1],[0 0 0 0 0 0]] for each x,y,1

		A1 = It1_col_grad*col
		A2 = It1_col_grad*row
		A3 = It1_col_grad
		A4 = It1_row_grad*col
		A5 = It1_row_grad*row
		A6 = It1_row_grad
		A = np.stack((A1,A2,A3,A4,A5,A6),axis=1)

		#2.calculate b
		#subtract It-It1_warped
		b = It_cont.ev(row,col)-It1_cont.ev(new_row,new_col)

		#3.perform lstsq to get delta_p and update delta_p
		delta_p,_,rank,_ = np.linalg.lstsq(A,b,rcond=None)

		#4.update p
		# print('before update')
		# print(p)
		# print(delta_p.flatten())
		p = p + delta_p.flatten()
		# print('after update')
		# print(p)
		M[0,0]=1+p[0]
		M[0,1]=p[1]
		M[0,2]=p[2]
		M[1,0]=p[3]
		M[1,1]=1+p[4]
		M[1,2]=p[5]
		# print('------------------------------------')
		# print(i)
		# print('Norm')
		# print(np.linalg.norm(delta_p.flatten()))
		# print('M')
		# print(M)
		# print(np.linalg.norm(delta_p.flatten()))
		# if(i>50):
		# 	break
		#     print(M)
	return M

