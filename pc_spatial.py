#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 12:19:01 2025

@author: prateeksrivastava
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from utilities import circ  

# Constants
lambda_ = 650e-9  # center wavelength (m)

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
f = 0.25  # Fraunhofer focal distance (m)
lf = lambda_ * f

# Partial spatial coherence screen parameters
N = 100  # number of screens (even)
Lcr = 8e-3  # spatial correlation length (m)
sigma_f = 2.5 * Lcr  # Gaussian filter parameter
sigma_r = np.sqrt(4 * np.pi * sigma_f**4 / Lcr**2)  # random std

dfx1 = 1 / L1
fx1 = np.linspace(-1/(2*dx1), 1/(2*dx1) - dfx1, M)
fx1 = fftshift(fx1)
FX1, FY1 = np.meshgrid(fx1, fx1)

# Source field
u1 = (
    circ(np.sqrt((X1 - dels/2)**2 + Y1**2) / w) +
    circ(np.sqrt((X1 + dels/2)**2 + Y1**2) / w)
)

# Filter spectrum
F = np.exp(-np.pi**2 * sigma_f**2 * (FX1**2 + FY1**2))

# Initialize irradiance array
I2 = np.zeros((M, M))

# Loop through screens
for n in range(N // 2):
    # Make two random screens
    fie = (ifft2(F * (np.random.randn(M) + 1j * np.random.randn(M))) *
           sigma_r / dfx1) * M**2 * dfx1**2
    
    # Fraunhofer pattern applying screen 1
    u2 = (1 / lf) * fft2(u1 * np.exp(1j * np.real(fie))) * dx1**2
    I2 += np.abs(u2)**2
    
    # Fraunhofer pattern applying screen 2
    u2 = (1 / lf) * fft2(u1 * np.exp(1j * np.imag(fie))) * dx1**2
    I2 += np.abs(u2)**2

# Normalize and center irradiance
I2 = ifftshift(I2) / N

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

# Plot irradiance slice
plt.figure(2)
plt.plot(x2, I2[M//2, :])
plt.xlabel('x (m)')
plt.ylabel('Irradiance')
plt.title('Irradiance Profile')

plt.show()
