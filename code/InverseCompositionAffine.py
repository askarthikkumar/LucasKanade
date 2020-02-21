import numpy as np
from scipy.interpolate import RectBivariateSpline

def get_mask(row, col, m , n):
	row_positive = row>=0
	col_positive = col>=0
	row_upper = row<m
	col_upper = col<n
	row_mask = np.logical_and(row_positive,row_upper)
	col_mask = np.logical_and(col_positive,col_upper)
	return np.logical_and(row_mask,col_mask)

def InverseCompositionAffine(It, It1):
	# Input: 
	#	It: template image
	#	It1: Current image

	# Output:
	#	M: the Affine warp matrix [2x3 numpy array]

    # put your implementation here

	M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],[0.0,0.0,1.0]])


	col = np.arange(0,It.shape[1])
	row = np.arange(0,It.shape[0])
	It_cont = RectBivariateSpline(row,col,It)
	It1_cont = RectBivariateSpline(row,col,It1)

	#scipy.interpolate uses array-like indexing so dx is actually dy in cartesian
	row_coords,col_coords = np.meshgrid(row,col)
	col_coords = col_coords.flatten()
	row_coords = row_coords.flatten()

	m,n = np.shape(It)

	delta_p=np.ones((1,6))


	It_col_grad = It_cont.ev(row_coords,col_coords,dx=0,dy=1)
	It_row_grad = It_cont.ev(row_coords,col_coords,dx=1,dy=0)

	#Multiply It' warped with W'
	#warp jacobian is [[0 0 1 0 0 0],[0 0 0 0 0 1],[0 0 0 0 0 0]] for each x,y,1

	A1 = It_col_grad*col_coords
	A2 = It_col_grad*row_coords
	A3 = It_col_grad
	A4 = It_row_grad*col_coords
	A5 = It_row_grad*row_coords
	A6 = It_row_grad
	A = np.stack((A1,A2,A3,A4,A5,A6),axis=1)
	# print(A)
	i=0
	while np.linalg.norm(delta_p.flatten())>1e-2:

		coords = M[:2,:2]@np.stack((col_coords,row_coords),axis=0)+np.reshape(M[:2,2],(2,1))
		# print(coords)
		new_col_coords = coords[0]
		new_row_coords = coords[1]
		# mask = np.logical_and(np.logical_and(new_col_coords>=0,new_row_coords>=0),np.logical_and(new_col_coords<n,new_row_coords<m))
		mask = get_mask(new_row_coords,new_col_coords,m,n)
		new_col = new_col_coords[mask]
		new_row = new_row_coords[mask]

		col = col_coords[mask]
		row = row_coords[mask]

		#1.calculate A
		A_ = A[mask]

		#2.calculate b
		#subtract It-It1_warped
		b = It1_cont.ev(new_row,new_col)-It_cont.ev(row,col)

		#3.perform lstsq to get delta_p and update delta_p
		delta_p,_,rank,_ = np.linalg.lstsq(A_,b,rcond=None)
		#     np.dot(np.linalg.inv(np.matmul(np.transpose(A_valid), A_valid)), b_)

		#4.update M
		delta_p_ = np.reshape(delta_p,(2,3))
		delta_p_ = np.vstack((delta_p_,np.ones((1,3))))
		#     print(i)
		delta_p_[0,0]+=1
		delta_p_[1,1]+=1
		delta_p_[2,0]=delta_p_[2,1]=0
		M = M@np.linalg.inv(delta_p_)
    #dont need to flip because calculated for (col,row) pairs.
	return M[:2,:]
