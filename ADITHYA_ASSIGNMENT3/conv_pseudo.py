import numpy as np


def conv2d(tensor1,tensor2):
	if len(tensor2.shape)>2:
		w1,h1,d1=tensor1.shape
		w2,h2,d2=tensor2.shape
		out_tensor=np.zeros([w1-w2+1,h1-h2+1])
		for i in range(out_tensor.shape[0]):
			for j in range(out_tensor.shape[1]):
				macc=0
				for k in range(w2):
					for l in range(h2):
						for d in range(d1): 
							macc+=tensor1[i+k][j+l][d]*tensor2[k][l][d]
				out_tensor[i][j]=macc

		return out_tensor
	else:
		w1,h1=tensor1.shape
		w2,h2=tensor2.shape
		out_tensor=np.zeros([w1-w2+1,h1-h2+1])
		for i in range(out_tensor.shape[0]):
			for j in range(out_tensor.shape[1]):
				macc=0
				for k in range(w2):
					for l in range(h2): 
						macc+=tensor1[i+k][j+l]*tensor2[k][l]
				out_tensor[i][j]=macc
		return out_tensor		
					
def stride_conv2d(tensor1,tensor2,stride):
	w1,h1,d1=tensor1.shape
	w2,h2,d2=tensor2.shape
	out_tensor=np.zeros([(w1-w2)//stride+1,(h1-h2)//stride+1])
	for i in range(out_tensor.shape[0]):
		for j in range(out_tensor.shape[1]):
			macc=0
			for k in range(w2):
				for l in range(h2):
					for d in range(d1): 
						macc+=tensor1[stride*i+k][stride*j+l][d]*tensor2[k][l][d]
			out_tensor[i][j]=macc

		return out_tensor



def dilated_conv2d(tensor1,tensor2,dilation=1):
	w1,h1,d1=tensor1.shape
	w2,h2,d2=tensor2.shape
	out_tensor=np.zeros([w1-(w2+(dilation-1)*(w2-1))+1,h1-(h2+(dilation-1)*(h2-1))+1])
	for i in range(out_tensor.shape[0]):
		for j in range(out_tensor.shape[1]):
			macc=0
			for k in range(w2):
				for l in range(h2):
					for d in range(d1): 
						macc+=tensor1[i+k*dilation][j+l*dilation][d]*tensor2[k][l][d]
			out_tensor[i][j]=macc

	return out_tensor	 	

def fractional_strided_conv(tensor1,tensor2,inv_stride):
	#inv_stride=1/fractional stride
	# for non-padded convolutions, we must pad tensor1
	w1,h1,d1=tensor1.shape
	#dilate the tensor 1
	new_matrix=np.zeros([w1+(inv_stride-1)*(w1-1),h1+(inv_stride-1)*(h1-1),d1])
	#copy tensor1->new_matrix
	for w in range(w1):
		for h in range(h1):
			for d in range(d1):
				new_matrix[w*inv_stride,h*inv_stride,d]=tensor1[w,h,d]
	#then apply conv2d as usual
	out_tensor=conv2d(new_matrix,tensor2)
	return out_tensor


def seperable_conv(tensor1,tensor2):
	w1,h1=tensor1.shape
	w2,h2=tensor2.shape
	[U,S,V]=np.linalg.svd(tensor2)
	filter1=np.array([U.T[0]])*S[0]
	filter2=np.array([V[0]])
	filter2=filter2.T
	out_tensor=conv2d(conv2d(tensor1,filter1),filter2)
	return out_tensor


def depthwise_seperable_conv(tensor1,tensor2,tensor3):
	w1,h1,d1=tensor1.shape
	w2,h2,d2=tensor2.shape
	arr_1=[]s
	for i in range(d1):
		arr_1.append(conv2d(tensor1[:,:,i],tensor2[:,:,i]))
	intermediate_mat=np.stack(arr_1,axis=2)
	return conv2d(intermediate_mat,tensor3)

t1=np.random.binomial(1,0.5,size=[5,5,2])
t2=np.random.binomial(1,0.5,size=[3,3,2])
t3=np.random.binomial(1,0.5,size=[1,1,2])
print("input tensors t1,t2,t3{}{}{}".format(t1,t2,t3))


print("2D convolution ")
print(conv2d(t1,t2))

print("2D convolution stride 2")
print(stride_conv2d(t1,t2,2))

print("2D Dilated convolution dilation 2")
print(dilated_conv2d(t1,t2,2))

print("Transpose convolution stride 0.5 (same padding)")
print(fractional_strided_conv(t2,t1,2))

print("Seperable convolution using SVD")
print(seperable_conv(t1[:,:,0],t2[:,:,0]))

print("Depthwise Seperable COnvolution")
print(depthwise_seperable_conv(t1,t2,t3))