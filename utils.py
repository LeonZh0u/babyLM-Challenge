import torch
import numpy as np

def log_domain_matmul(log_A, log_B):
	"""
	log_A : m x n
	log_B : n x p
	output : m x p matrix

	Normally, a matrix multiplication
	computes out_{i,j} = sum_k A_{i,k} x B_{k,j}

	A log domain matrix multiplication
	computes out_{i,j} = logsumexp_k log_A_{i,k} + log_B_{k,j}
	"""
	m = log_A.shape[0]
	n = log_A.shape[1]
	p = log_B.shape[1]

	# log_A_expanded = torch.stack([log_A] * p, dim=2)
	# log_B_expanded = torch.stack([log_B] * m, dim=0)
    # fix for PyTorch > 1.5 by egaznep on Github:
	log_A_expanded = torch.reshape(log_A, (m,n,1))
	log_B_expanded = torch.reshape(log_B, (1,n,p))

	elementwise_sum = log_A_expanded + log_B_expanded
	out = torch.logsumexp(elementwise_sum, dim=1)

	return out

def maxmul(log_A, log_B):
	"""
	log_A : m x n
	log_B : n x p
	output : m x p matrix

	Similar to the log domain matrix multiplication,
	this computes out_{i,j} = max_k log_A_{i,k} + log_B_{k,j}
	"""
	m = log_A.shape[0]
	n = log_A.shape[1]
	p = log_B.shape[1]

	log_A_expanded = torch.stack([log_A] * p, dim=2)
	log_B_expanded = torch.stack([log_B] * m, dim=0)

	elementwise_sum = log_A_expanded + log_B_expanded
	out1,out2 = torch.max(elementwise_sum, dim=1)

	return out1,out2