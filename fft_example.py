#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 22:24:06 2025

@author: prateeksrivastava
"""

import numpy as np
import matplotlib.pyplot as plt
from utilities import rect  

# Parameters
w = 0.055  # rectangle half-width (m)
L = 2      # vector side length (m)
M = 200    # number of samples
dx = L / M  # sample interval (m)

# Coordinate vector
x = np.linspace(-L/2, L/2 - dx, M)  # equivalent to -L/2:dx:L/2-dx
f = rect(x / (2 * w))  # signal vector

# Plotting
plt.figure(1)
plt.plot(x, f)  # plot f vs x
plt.title('Signal vs x')
plt.xlabel('x (m)')
plt.ylabel('f')

plt.figure(2)
plt.plot(x, f, '-o')  # plot f vs x with markers
plt.axis([-0.2, 0.2, 0, 1.5])
plt.xlabel('x (m)')
plt.title('Signal vs x with markers')

plt.figure(3)
plt.plot(f, '-o')
plt.axis([80, 120, 0, 1.5])
plt.xlabel('index')
plt.title('Signal values')

# FFT and scaling
f0 = np.fft.fftshift(f)  # shift f

plt.figure(4)
plt.plot(f0)
plt.axis([0, 200, 0, 1.5])
plt.xlabel('index')
plt.title('Shifted Signal')

F0 = np.fft.fft(f0) * dx  # FFT and scale

# Magnitude Plot
plt.figure(5)
plt.plot(np.abs(F0))  # plot magnitude
plt.title('Magnitude')
plt.xlabel('index')

# Phase Plot
plt.figure(6)
plt.plot(np.angle(F0))  # plot phase
plt.title('Phase')
plt.xlabel('index')

F = np.fft.fftshift(F0)  # center F
fx = np.linspace(-1/(2*dx), 1/(2*dx) - (1/L), M)  # frequency coordinates

# Magnitude in frequency domain
plt.figure(7)
plt.plot(fx, np.abs(F))  # plot magnitude
plt.title('Magnitude in Frequency Domain')
plt.xlabel('fx (cyc/m)')

# Phase in frequency domain
plt.figure(8)
plt.plot(fx, np.angle(F))  # plot phase
plt.title('Phase in Frequency Domain')
plt.xlabel('fx (cyc/m)')

# Analytic result
F_an = 2 * w * np.sinc(2 * w * fx)  # analytic result

# Comparison of discrete and analytic magnitude
plt.figure(9)
plt.plot(fx, np.abs(F), label='discrete', color='b')
plt.plot(fx, np.abs(F_an), ':', label='analytic', color='r')  
plt.title('Magnitude Comparison')
plt.legend()
plt.xlabel('fx (cyc/m)')

# Comparison of discrete and analytic phase
plt.figure(10)
plt.plot(fx, np.angle(F), label='discrete', color='b')
plt.plot(fx, np.angle(F_an), ':', label='analytic', color='r')  
plt.title('Phase Comparison')
plt.legend()
plt.xlabel('fx (cyc/m)')

# Show all figures
plt.show()
