import numpy as np

def get_XS(x,s):
	X = np.diag(x)
	S = np.diag(s)
	e = np.ones(x.shape)
	# XSe = np.matmul(np.matmul(X,S),e)
	return X, S

def make_A_dash(A,S,X,nv,ne):
	z1 = np.zeros((nv,nv))
	z2 = np.zeros((nv,ne))
	row1 = np.concatenate((A,z1),axis=1)
	row1 = np.concatenate((row1,z2),axis=1)
	# print(row1.shape)

	z3 = np.zeros((ne,ne))
	I = np.eye(ne)
	row2 = np.concatenate((z3,A.T,),axis=1)
	row2 = np.concatenate((row2,I),axis=1)
	# print(row2.shape)

	row3 = np.concatenate((S,z2.T),axis=1)
	row3 = np.concatenate((row3,X),axis=1)
	# print(row3.shape)

	M = np.concatenate((row1,row2),axis=0)
	M = np.concatenate((M,row3),axis=0)

	return M

def make_b_dash(A,b,c,x,y,s,mu,nv,ne):
	e = np.ones(ne)
	X,S = get_XS(x,s)
	XSe = np.matmul(np.matmul(X,S),e)

	sigma = 0.2

	row1 = b - np.matmul(A,x)
	row2 = c - np.matmul(A.T,y) - s
	row3 = sigma*mu*e - XSe

	b = np.concatenate((row1,row2),axis=None)
	b = np.concatenate((b,row3),axis=None)

	return b

def get_best_alpha(x,s,del_x,nv,ne,alpha):
	x_dash = del_x[:ne]
	y_dash = del_x[ne:nv+ne]
	s_dash = del_x[nv+ne:]

	alpha_x = []
	alpha_s = []

	# print(s,s_dash)
	# print(x,x_dash)

	for i in range(x.shape[0]):
		if x_dash[i]<0:
			alpha_x.append(x[i]/-x_dash[i])
		if s_dash[i]<0:
			alpha_s.append(s[i]/-s_dash[i])

	if len(alpha_x)==0 and len(alpha_s)==0:
		return alpha
	else:
		alpha_x.append(np.inf)
		alpha_s.append(np.inf)
		alpha_x = np.array(alpha_x)
		alpha_s = np.array(alpha_s)

		alpha_max = min(np.min(alpha_x), np.min(alpha_s))

		eta = 0.999
		alpha_k = min(1,eta*alpha_max)

	return alpha_k


def interior_point(A,b,c,nv,ne):
	x = np.ones(ne)
	y = np.ones(nv)
	s = np.ones(ne)

	alpha = 0.5
	epsilon = 1e-10

	while(True):
		mu = np.dot(x,s)/ne
		X,S = get_XS(x,s)
		A_dash = make_A_dash(A,S,X,nv,ne)
		b_dash = make_b_dash(A,b,c,x,y,s,mu,nv,ne)

		print('A_dash',A_dash)
		if mu<epsilon or x.all()<0 or s.all()<0:
			print(x)
			# print(s)
			print(np.dot(c.T,x))
			return x

		del_x = np.linalg.solve(A_dash,b_dash)

		alpha = get_best_alpha(x,s,del_x,nv,ne,alpha)

		x = x + alpha*del_x[:ne]
		y = y + alpha*del_x[ne:ne+nv]
		s = s + alpha*del_x[ne+nv:]

		# print(alpha)
# if __name__ == '__main__':

# 	A = np.array([[1,1,1,0],[3,2,0,1]])
# 	b = np.array([5,12])
# 	# b = b.reshape(b.shape[0],1)
# 	c = np.array([-6,-5,0,0])
# 	x = np.zeros(A.shape[1])

# 	interior_point(A,b,c,A.shape[0],A.shape[1])