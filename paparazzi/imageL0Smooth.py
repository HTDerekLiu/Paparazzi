## Referneces:
## 1. Xu, "Image Smoothing via L0 Gradient Minimization", 2011
## 2. https://github.com/t-suzuki/l0_gradient_minimization_test

import numpy as np
from scipy.fftpack import fft2, ifft2
import matplotlib.pyplot as plt

def imageL0Smooth(I, lmd = 0.05):
	# I: original image
	# S: output image

	betaMax = 1e5
	beta = 0.1
	betaRate = 2.0
	numIter = 40

	# compute FFT denominator (second part only)
	FI = fft2(I, axes=(0, 1))
	dx = np.zeros((I.shape[0], I.shape[1])) # gradient along x direction
	dy = np.zeros((I.shape[0], I.shape[1])) # gradient along y direction
	dx[dx.shape[0]/2, dx.shape[1]/2-1:dx.shape[1]/2+1] = [-1,1]
	dy[dy.shape[0]/2-1:dy.shape[0]/2+1, dy.shape[1]/2] = [-1,1]
	denominator_second = np.conj(fft2(dx))*fft2(dx) + np.conj(fft2(dy))*fft2(dy)
	denominator_second = np.tile(np.expand_dims(denominator_second, axis=2), [1,1,I.shape[2]])

	S = I 
	hp = 0*I
	vp = 0*I
	for iter in range(numIter):
		# solve hp, vp
		hp = np.concatenate((S[:,1:], S[:,:1]), axis=1) - S
		vp = np.concatenate((S[1:,:], S[:1,:]), axis=0) - S
		if len(I.shape) == 3:
			zeroIdx = np.sum(hp**2+vp**2, axis=2) < lmd/beta
		else:
			zeroIdx = hp**2.0 + vp**2.0 < lmd/beta
		hp[zeroIdx] = 0.0
		vp[zeroIdx] = 0.0

		# solve S
		hv = np.concatenate((hp[:,-1:], hp[:,:-1]), axis=1) - hp + np.concatenate((vp[-1:,:], vp[:-1,:]), axis=0) - vp
		S = np.real(ifft2((FI + (beta*fft2(hv, axes=(0, 1)))) / (1+beta*denominator_second), axes=(0, 1)))

		# update parameters
		beta *= betaRate
		if beta > betaMax: 
			break

		# plt.imshow(S)
		# plt.pause(0.01)
	return S
