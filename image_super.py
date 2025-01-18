#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 12:04:22 2025

@author: prateeksrivastava
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from PIL import Image
from utilities import seidel_5, circ

# Load and preprocess image
A = np.array(Image.open('sampleImage.png').convert('L'))
M, N = A.shape
A = np.flipud(A)
Ig = A.astype(float) / A.max()

# Set up parameters
L = 1e-3  # image plane side length
du = L / M  # sample interval
u = v = np.arange(-L/2, L/2, du)  # coordinates
fN = 1 / (2 * du)  # Nyquist frequency

lambda_ = 0.5 * 10**-6  # wavelength
k = 2 * np.pi / lambda_  # wavenumber
wxp = 2.5e-3  # exit pupil radius
zxp = 100e-3  # exit pupil distance
fnum = zxp / (2 * wxp)  # exit pupil f-number

twof0 = 1 / (lambda_ * fnum)  # inc cutoff freq

# Aberration coefficients
wd = 0 * lambda_
w040 = 0.5 * lambda_
w131 = 1 * lambda_
w222 = 1.5 * lambda_
w220 = 0 * lambda_
w311 = 0 * lambda_

# Reverse image plane frequency coordinates
fu = np.fft.fftshift(np.fft.fftfreq(M, du))[::-1]
Fu, Fv = np.meshgrid(fu, fu)

I = np.zeros((M, M))

# Loop through image plane positions
for n in range(M):
    v0 = (n - (M/2 + 1)) / (M/2)  # norm v image coord
    print(f"Processing row {n+1}/{M}")  # Progress indicator
    for m in range(M):
        u0 = (m - (M/2 + 1)) / (M/2)  # norm u image coord
        
        # Wavefront
        W = seidel_5(u0, v0, -2*lambda_*fnum*Fu, -2*lambda_*fnum*Fv,
                     wd, w040, w131, w222, w220, w311)
        
        # Coherent transfer function
        H = circ(np.sqrt(Fu**2 + Fv**2) * 2 * lambda_ * fnum) * np.exp(-1j * k * W)
        
        # PSF
        h2 = np.abs(ifftshift(ifft2(H)))**2
        
        # Shift h2 to image plane position
        h2 = np.roll(h2, (n - (M//2 + 1), m - (M//2 + 1)), axis=(0, 1))
        
        # Superposition integration
        I[n, m] = np.sum(Ig * h2)

# Plot the result
plt.figure(figsize=(8, 8))
plt.imshow(np.power(I, 1/3), extent=[u.min(), u.max(), v.min(), v.max()], 
           cmap='gray', origin='lower')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('Superposition Image')
plt.colorbar()
plt.axis('square')
plt.show()
