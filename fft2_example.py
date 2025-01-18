#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 22:29:32 2025

@author: prateeksrivastava
"""


import numpy as np
import matplotlib.pyplot as plt
from utilities import rect  

# Define parameters
wx = 0.1  # rect x half-width (m)
wy = 0.05  # rect y half-width (m)
L = 2  # side length x&y (m)
M = 200  # samples/side length
dx = L / M  # sample interval (m)

# Create arrays
x = np.linspace(-L/2, L/2 - dx, int(L/dx))
y = x
[X, Y] = np.meshgrid(x, y)
g = rect(X/(2*wx)) * rect(Y/(2*wy))

# Display image
plt.figure(figsize=(8, 8))
plt.imshow(g, extent=[-L/2, L/2, -L/2, L/2], cmap='gray')
plt.axis('equal')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('Rectangular Signal')

# Plot x-slice profile
plt.figure()
plt.plot(x, g[M//2 + 1, :])
plt.xlabel('x (m)')
plt.ylabel('Amplitude')
plt.title('x-slice Profile')
plt.axis([-1, 1, 0, 1.2])

# Perform FFT
g0 = np.fft.ifftshift(g)
G0 = np.fft.fft2(g0) * (dx**2)
G = np.fft.ifftshift(G0)

# Calculate frequency coordinates
fx = np.linspace(-1/(2*dx), 1/L, len(x))
fy = fx

# Create surface plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(fx[:, None], fy[None, :], np.abs(G),
                       cmap='viridis', edgecolor='none')

plt.xlabel('fx (cyc/m)')
plt.ylabel('fy (cyc/m)')
plt.title('Magnitude Spectrum (Surface Plot)')
plt.colorbar(surf, shrink=0.7, aspect=30)

# Plot frequency domain x-slice profile
plt.figure()
plt.plot(fx, np.abs(G[M//2 + 1, :]))
plt.xlabel('fx (cyc/m)')
plt.ylabel('|G|')
plt.title('Frequency Domain x-slice Profile')

plt.show()


