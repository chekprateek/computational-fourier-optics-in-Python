#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 22:27:18 2025

@author: prateeksrivastava
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
wa = 0.3  # Gaussian 1 width [exp(-pi) radius] (m)
wb = 0.2  # Gaussian 2 width [exp(-pi) radius] (m)
L = 2     # side length (meters)
M = 200   # number of samples
dx = L / M  # sample interval (m)

# x coordinates
x = np.linspace(-L/2, L/2 - dx, M)  # equivalent to [-L/2:dx:L/2-dx]

# Gaussian functions
fa = np.exp(-np.pi * (x**2) / wa**2)  # Gaussian a
fb = np.exp(-np.pi * (x**2) / wb**2)  # Gaussian b

# Plotting the functions
plt.figure(1)
plt.plot(x, fa, label='Gaussian a')
plt.plot(x, fb, '--', label='Gaussian b')
plt.title('Functions')
plt.xlabel('x (m)')
plt.legend()

# FFT of the functions
Fa = np.fft.fft(fa)  # transform fa
Fb = np.fft.fft(fb)  # transform fb

# Pointwise multiplication in frequency domain
F0 = Fa * Fb  

# Inverse FFT and scaling
f0 = np.fft.ifft(F0) * dx  # inverse transform and scale
f = np.fft.fftshift(f0)     # center result

# Plotting the convolution result
plt.figure(2)
plt.plot(x, f.real)  # Plot only the real part of the convolution result
plt.title('Convolution')
plt.xlabel('x (m)')
plt.grid()

# Show all figures
plt.show()
