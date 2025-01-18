#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 22:41:51 2025

@author: prateeksrivastava
"""

import numpy as np
import matplotlib.pyplot as plt
from utilities import jinc  # Import the jinc function from utilities.py

# Parameters
L = 0.2          # side length (m)
M = 250          # number of samples
dx = L / M       # sample interval
x = np.linspace(-L/2, L/2 - dx, M)  # x coordinates
y = x.copy()     # y coordinates
X, Y = np.meshgrid(x, y)  # Create meshgrid for X and Y

# Additional parameters
w = 1e-3         # x half-width (m)
lambda_ = 0.633e-6  # wavelength (m)
z = 50           # propagation distance (m)
k = 2 * np.pi / lambda_  # wavenumber
lz = lambda_ * z  # propagation factor

# Irradiance calculation
I2 = (w**2 / lz)**2 * (jinc(w / lz * np.sqrt(X**2 + Y**2)))**2

# Plotting the irradiance image
plt.figure(1)  # Irradiance image
plt.imshow(np.cbrt(I2), extent=(-L/2, L/2, -L/2, L/2), cmap='gray', origin='lower')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.axis('square')
plt.title('Fraunhofer Irradiance')
plt.colorbar(label='Irradiance')

# Plotting the x-axis profile
plt.figure(2)  # x-axis profile
plt.plot(x, I2[M//2, :])  # Profile at the middle of the y-axis
plt.xlabel('x (m)')
plt.ylabel('Irradiance')
plt.title('Irradiance Profile along x-axis')

# Show all figures
plt.show()
