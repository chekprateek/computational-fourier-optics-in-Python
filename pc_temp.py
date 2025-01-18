#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 12:14:25 2025

@author: prateeksrivastava
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from PIL import Image
from utilities import circ  # Importing the circ function from your utilities.py

# Constants
lambda0 = 650e-9  # center wavelength (m)
c = 3e8  # speed of light (m/s)
k0 = 2 * np.pi / lambda0  # center wavenumber
nu0 = c / lambda0  # center frequency

# Gaussian lineshape parameters
N = 51  # number of components (odd)
delnu = 2e9  # spectral density FWHM (Hz)
b = delnu / (2 * np.sqrt(np.log(2)))  # FWHM scaling
dnu = 4 * delnu / N  # frequency interval

# Source plane parameters
L1 = 50e-3  # source plane side length (m)
M = 250  # number of samples (even)
dx1 = L1 / M  # sample interval
x1 = np.linspace(-L1/2, L1/2 - dx1, M)  # source coords
x1 = fftshift(x1)  # shift x coord
X1, Y1 = np.meshgrid(x1, x1)

# Beam parameters
w = 1e-3  # radius (m)
dels = 5e-3  # transverse separation (m)
deld = 5e-2  # delay distance (m)
f = 0.25  # focal distance for Fraunhofer
lf = lambda0 * f

# Initialize irradiance array
I2 = np.zeros((M, M))

# Loop through lines
for n in range(N):
    # Spectral density function
    nu = (n - (N + 1) / 2) * dnu + nu0
    S = 1 / (np.sqrt(np.pi) * b) * np.exp(-(nu - nu0)**2 / b**2)
    k = 2 * np.pi * nu / c
    
    # Source field calculation
    u = (
        circ(np.sqrt((X1 - dels/2)**2 + Y1**2) / w) +
        circ(np.sqrt((X1 + dels/2)**2 + Y1**2) / w) *
        np.exp(1j * k * deld)
    )
    
    # Fraunhofer pattern
    u2 = (1 / lf) * fft2(u) * dx1**2
    
    # Weighted irradiance and sum
    I2 += S * (np.abs(u2)**2) * dnu

# Normalize and center irradiance
I2 = ifftshift(I2)

# Observation coordinates
x2 = np.linspace(-1/(2*dx1), 1/(2*dx1) - 1/L1, M) * lf
y2 = x2

# Plot irradiance image
plt.figure(1)
plt.imshow(I2, extent=(x2.min(), x2.max(), y2.min(), y2.max()), cmap='gray', origin='lower')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.axis('square')
plt.title('Irradiance Image')

# Plot irradiance profile
plt.figure(2)
plt.plot(x2, I2[M//2, :], '-o')
plt.xlabel('x (m)')
plt.ylabel('Irradiance')
plt.title('Irradiance Profile')

plt.show()
