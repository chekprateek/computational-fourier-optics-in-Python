#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 11:57:06 2025

@author: prateeksrivastava
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from utilities import seidel_5, circ

# Lens and simulation parameters
M = 1024  # sample #
L = 1e-3  # image plane side length
du = L / M  # sample interval
u = v = np.arange(-L/2, L/2, du)  # coordinates

lambda_ = 0.55e-6  # wavelength
k = 2 * np.pi / lambda_  # wavenumber
Dxp = 20e-3
wxp = Dxp / 2  # exit pupil size
zxp = 100e-3  # exit pupil distance
fnum = zxp / (2 * wxp)  # exit pupil f-number
lz = lambda_ * zxp
twof0 = 1 / (lambda_ * fnum)  # incoh cutoff freq

u0, v0 = 0, 0  # normalized image coordinate

# Aberration coefficients
wd = 0 * lambda_
w040 = 4.963 * lambda_
w131 = 2.637 * lambda_
w222 = 9.025 * lambda_
w220 = 7.536 * lambda_
w311 = 0.157 * lambda_

fu = np.fft.fftshift(np.fft.fftfreq(M, du))
Fu, Fv = np.meshgrid(fu, fu)

# Wavefront
W = seidel_5(u0, v0, -lz*Fu/wxp, -lz*Fv/wxp, wd, w040, w131, w222, w220, w311)

# Coherent transfer function
H = circ(np.sqrt(Fu**2 + Fv**2) * lz / wxp) * np.exp(-1j * k * W)

plt.figure(1)
plt.imshow(np.angle(H), extent=[u.min(), u.max(), v.min(), v.max()], cmap='gray')
plt.xlabel('u (m)'); plt.ylabel('v (m)')
plt.axis('square')

# Point spread function
h2 = np.abs(ifftshift(ifft2(fftshift(H))))**2

plt.figure(2)
plt.imshow(np.power(h2, 1/2), extent=[u.min(), u.max(), v.min(), v.max()], cmap='gray')
plt.xlabel('u (m)'); plt.ylabel('v (m)')
plt.axis('square')

plt.figure(3)
plt.plot(u, h2[M//2, :])
plt.xlabel('u (m)'); plt.ylabel('PSF')

plt.figure(4)
plt.plot(u, h2[:, M//2])
plt.xlabel('v (m)'); plt.ylabel('PSF')

# MTF
MTF = fft2(fftshift(h2))
MTF = np.abs(MTF / MTF[0, 0])  # normalize DC to 1
MTF = ifftshift(MTF)

# Analytic MTF
MTF_an = (2/np.pi) * (np.arccos(fu/twof0) - (fu/twof0) * np.sqrt(1 - (fu/twof0)**2))
MTF_an = MTF_an * np.where(np.abs(fu) <= twof0, 1, 0)  # zero after cutoff

plt.figure(5)
plt.plot(fu, MTF[M//2, :], fu, MTF[:, M//2], ':', fu, MTF_an, '--')
plt.axis([0, 150000, 0, 1])
plt.legend(['u MTF', 'v MTF', 'diff limit'])
plt.xlabel('f (cyc/m)'); plt.ylabel('Modulation')

plt.show()
