#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 12:01:41 2025

@author: prateeksrivastava
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from utilities import seidel_5, circ

# Set up parameters
M = 250
L = 1e-3  # image plane side length
du = L / M  # sample interval
u = v = np.arange(-L/2, L/2, du)  # coordinates

lambda_ = 0.5 * 10**-6  # wavelength
k = 2 * np.pi / lambda_  # wavenumber
wxp = 2.5e-3  # exit pupil radius
zxp = 100e-3  # exit pupil distance
fnum = zxp / (2 * wxp)  # exit pupil f-number

twof0 = 1 / (lambda_ * fnum)  # inc cutoff freq
fN = 1 / (2 * du)  # Nyquist frequency

# Aberration coefficients
wd = 0 * lambda_
w040 = 0.5 * lambda_
w131 = 1 * lambda_
w222 = 1.5 * lambda_
w220 = 0 * lambda_
w311 = 0 * lambda_

fu = np.fft.fftshift(np.fft.fftfreq(M, du))  # image freq coords
Fu, Fv = np.meshgrid(fu, fu)

I = np.zeros((M, M))

# Loop through image plane positions
for u0 in np.arange(-0.7, 0.7 + 0.7/3, 0.7/3):
    for v0 in np.arange(-0.7, 0.7 + 0.7/3, 0.7/3):
        # Wavefront
        W = seidel_5(u0, v0, -2*lambda_*fnum*Fu, -2*lambda_*fnum*Fv,
                     wd, w040, w131, w222, w220, w311)
        
        # Coherent transfer function
        H = circ(np.sqrt(Fu**2 + Fv**2) * 2 * lambda_ * fnum) * np.exp(-1j * k * W)
        
        # PSF
        h2 = np.abs(ifftshift(ifft2(H)))**2
        
        # Shift PSF to image plane position
        h2 = np.roll(h2, (round(v0*M/2), round(u0*M/2)), axis=(0, 1))
        
        # Add into combined frame
        I += h2

# Plot the result
plt.figure(figsize=(8, 8))
plt.imshow(np.power(I, 1/2), extent=[u.min(), u.max(), v.min(), v.max()], 
           cmap='gray', origin='lower')
plt.xlabel('u (m)')
plt.ylabel('v (m)')
plt.title('PSF Map')
plt.colorbar()
plt.axis('square')
plt.show()
