import numpy as np
import interior_point as point

def make_A(M,u,nv,ne):
	k=0
	A = np.zeros((nv+ne,2*ne))
	for i in range(nv):
		for j in range(nv):
			if(M[i][j]!=0):
				A[i][k]=-1
				A[j][k]=1
				k+=1

	k = 0
	for i in range(nv,nv+ne):
		A[i][k] = 1
		A[i][k+ne] = 1
		k+=1

	A = np.delete(A,0,0)
	A = np.delete(A,nv-2,0)
	return(A)


def make_u(M):
	c = []
	for i in range(len(M)):
		for j in range(len(M)):
			if(M[i][j]!=0):
				c.append(M[i][j])
	return(np.array(c))


if __name__ == '__main__':
	# M = [[0.,16.,13.,0.,0.,0.],[0.,0.,10.,12.,0.,0.],[0.,4.,0.,0.,14.,0.],[0.,0.,9.,0.,0.,20.],[0.,0.,0.,7.,0.,7.],[0.,0.,0.,0.,0.,0.]]
	M = [[0,11,15,10,0,0,0,0,0,0,0,0],[0,0,0,0,0,18,4,0,0,0,0,0],[0,3,8,5,0,0,0,0,0,0,0,0],[0,0,0,0,6,0,0,3,11,0,0,0],[0,0,0,4,0,0,0,17,6,0,0,0],[0,0,0,0,3,16,0,0,0,13,0,0],[0,12,0,0,4,0,0,0,0,0,0,21],[0,0,0,0,0,0,0,0,4,9,4,3],[0,0,0,0,0,0,0,4,0,0,5,4],[0,0,0,0,0,0,0,0,0,0,7,9],[0,0,0,0,0,0,0,0,2,0,0,15],[0,0,0,0,0,0,0,0,0,0,0,0]]
	
	M = np.array(M)

	num_vertices = len(M)
	num_edges = 0

	for i in range(len(M)):
		for j in range(len(M)):
			if(M[i][j]!=0):
				num_edges += 1

	u = make_u(np.copy(M))
	A = make_A(np.copy(M), np.copy(u), num_vertices,num_edges)

	
	c = np.zeros(2*num_edges)
	k=0
	for i in M[0]:
		if i!=0:
			c[k] = -1
			k+=1
	b = np.zeros(num_vertices-2)
	b = np.concatenate((b,u),axis=None)

	x = np.zeros(A.shape[1])

	point.interior_point(A,b,c,A.shape[0],A.shape[1])


