#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 11:48:04 2025

@author: prateeksrivastava
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from PIL import Image

# Read and preprocess image
A = np.array(Image.open('sampleImage.png').convert('L'))
M, N = A.shape
# A = np.flipud(A)
Ig = A.astype(float) / A.max()

ug = np.sqrt(Ig)  # ideal image field
L = 0.3e-3  # image plane side length (m)
du = L / M  # sample interval (m)
u = v = np.arange(-L/2, L/2, du)

# Plot ideal image
plt.figure(1)
plt.imshow(Ig, extent=[u.min(), u.max(), v.min(), v.max()], cmap='gray')
plt.xlabel('u (m)'); plt.ylabel('v (m)')
plt.axis('square')

# Set parameters
lambda_ = 0.5e-6  # wavelength
wxp = 6.25e-3  # exit pupil radius
zxp = 125e-3  # exit pupil distance
f0 = wxp / (lambda_ * zxp)  # cutoff frequency

fu = fv = np.fft.fftshift(np.fft.fftfreq(M, du))
Fu, Fv = np.meshgrid(fu, fv)

# Define circ function
def circ(r):
    return np.where(r <= 1, 1, 0)

H = circ(np.sqrt(Fu**2 + Fv**2) / f0)

# Plot H
plt.figure(2)
ax = plt.axes(projection='3d')
ax.plot_surface(Fu, Fv, H * 0.99, cmap='gray')
ax.set_xlabel('fu (cyc/m)'); ax.set_ylabel('fv (cyc/m)')

H = fftshift(H)
Gg = fft2(fftshift(ug))
Gi = Gg * H
ui = ifftshift(ifft2(Gi))
Ii = np.abs(ui)**2

# Plot image result
plt.figure(3)
plt.imshow(np.power(Ii, 1/2), extent=[u.min(), u.max(), v.min(), v.max()], cmap='gray')
plt.xlabel('u (m)'); plt.ylabel('v (m)')
plt.axis('square')

# Plot horizontal image slice
plt.figure(4)
vvalue = -0.8e-4
vindex = round(vvalue / du + (M / 2 + 1))
plt.plot(u, Ii[vindex, :], u, Ig[vindex, :], ':')
plt.xlabel('u (m)'); plt.ylabel('Irradiance')

plt.show()
