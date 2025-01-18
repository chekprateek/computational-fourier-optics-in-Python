#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 10:51:34 2025

@author: prateeksrivastava
"""

import numpy as np
import matplotlib.pyplot as plt
from utilities import propTF, propIR, propFF

# Square beam propagation example

L1 = 0.5  # side length
M = 250  # number of samples
dx1 = L1 / M  # src sample interval
x1 = np.arange(-L1/2, L1/2, dx1)  # src coords
y1 = x1
lambda_ = 0.5 * 10**-6  # wavelength
k = 2 * np.pi / lambda_  # wavenumber
w = 0.051  # source half width (m)
z = 2000  # propagation dist (m)

X1, Y1 = np.meshgrid(x1, y1)
u1 = np.logical_and(np.abs(X1) <= w, np.abs(Y1) <= w).astype(float)  # src field
I1 = np.abs(u1**2)  # src irradiance

# Figure 1
plt.figure(1)
plt.imshow(I1, extent=[x1.min(), x1.max(), y1.min(), y1.max()], cmap='gray')
plt.colorbar()
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('z = 0 m')
plt.axis('square')

# Select either propIR or propTF 
# u2 = propTF(u1, L1, lambda_, z)  # propagation
u2 = propIR(u1, L1, lambda_, z)  #propagation

x2 = x1  # obs coords
y2 = y1
I2 = np.abs(u2**2)  # obs irrad

# Figure 2
plt.figure(2)
plt.imshow(I2, extent=[x2.min(), x2.max(), y2.min(), y2.max()], cmap='gray')
plt.colorbar()
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title(f'z = {z} m')
plt.axis('square')

# Figure 3
plt.figure(3)
plt.plot(x2, I2[M//2, :])
plt.xlabel('x (m)')
plt.ylabel('Irradiance')
plt.title(f'z = {z} m')

# Figure 4
plt.figure(4)
plt.plot(x2, np.abs(u2[M//2, :]))
plt.xlabel('x (m)')
plt.ylabel('Magnitude')
plt.title(f'z = {z} m')

# Figure 5
plt.figure(5)
plt.plot(x2, np.unwrap(np.angle(u2[M//2, :])))
plt.xlabel('x (m)')
plt.ylabel('Phase (rad)')
plt.title(f'z = {z} m')

plt.show()


#%% Change w=0.011 above and then run following cell

w=0.011
X1, Y1 = np.meshgrid(x1, y1)
u1 = np.logical_and(np.abs(X1) <= w, np.abs(Y1) <= w).astype(float)  # src field
I1 = np.abs(u1**2)  # src irradiance


u2, L2 = propFF(u1, L1, lambda_, z)

# Calculate observation coordinates
dx2 = L2 / M
x2 = np.arange(-L2/2, L2/2, dx2)  # obs coords
y2 = x2

# Calculate intensity
I2 = np.abs(u2)**2

# Plot
plt.figure(1)
plt.imshow(np.power(I2, 1/3), extent=[x2.min(), x2.max(), y2.min(), y2.max()], 
           cmap='gray', origin='lower')
plt.colorbar()
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('Fraunhofer Diffraction Pattern')
plt.axis('square')
plt.show()

x2 = x1  # obs coords
y2 = y1
I2 = np.abs(u2**2)  # obs irrad


# Figure 3
plt.figure(2)
plt.plot(x2, I2[M//2, :], '-o')
plt.xlabel('x (m)')
plt.ylabel('Irradiance')
plt.title(f'z = {z} m')



plt.show()
