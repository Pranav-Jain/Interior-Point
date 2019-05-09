import numpy as np
import interior_point as point

def make_A_and_b(M,nv,ne):
	A = np.zeros((ne,2*(nv)+2*ne))
	b = np.zeros(ne)
	k=0
	for i in range(nv):
		for j in range(nv):
			if(M[i][j]!=0):
				if i==0:
					A[k][k]=1
					A[k][ne-1+j]=1
					A[k][ne-1+nv+j]=-1
					A[k][ne-1+2*(nv)+k]=-1
					b[k]=1
				elif j==nv-1:
					A[k][k]=1
					A[k][ne-1+i]=-1
					A[k][ne-1+nv+i]=1
					A[k][ne-1+2*(nv)+k]=-1
				else:
					A[k][k]=1
					A[k][ne-1+i]=-1
					A[k][ne-1+nv+i]=1
					A[k][ne-1+j]=1
					A[k][ne-1+nv+j]=-1
					A[k][ne-1+2*(nv)+k]=-1
				k+=1

	return(A,b)


def make_u(M):
	c = []
	for i in range(len(M)):
		for j in range(len(M)):
			if(M[i][j]!=0):
				c.append(M[i][j])
	return(np.array(c))


M = [[0.,16.,13.,0.,0.,0.],[0.,0.,10.,12.,0.,0.],[0.,4.,0.,0.,14.,0.],[0.,0.,9.,0.,0.,20.],[0.,0.,0.,7.,0.,7.],[0.,0.,0.,0.,0.,0.]]
# M = [[0,11,15,10,0,0,0,0,0,0,0,0],[0,0,0,0,0,18,4,0,0,0,0,0],[0,3,8,5,0,0,0,0,0,0,0,0],[0,0,0,0,6,0,0,3,11,0,0,0],[0,0,0,4,0,0,0,17,6,0,0,0],[0,0,0,0,3,16,0,0,0,13,0,0],[0,12,0,0,4,0,0,0,0,0,0,21],[0,0,0,0,0,0,0,0,4,9,4,3],[0,0,0,0,0,0,0,4,0,0,5,4],[0,0,0,0,0,0,0,0,0,0,7,9],[0,0,0,0,0,0,0,0,2,0,0,15],[0,0,0,0,0,0,0,0,0,0,0,0]]

M = np.array(M)

num_vertices = len(M)
num_edges = 0

for i in range(len(M)):
	for j in range(len(M)):
		if(M[i][j]!=0):
			num_edges += 1

A,b = make_A_and_b(np.copy(M),num_vertices,num_edges)

c = make_u(np.copy(M))
c = np.concatenate((c,np.zeros(2*(num_vertices)+num_edges)),axis=None)
# print(c)

point.interior_point(A,b,c,A.shape[0],A.shape[1])