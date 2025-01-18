#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 22:05:21 2025

@author: prateeksrivastava
"""

import numpy as np
from scipy.special import j1  # Bessel function of the first kind
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift

def circ(r):
    """
    Circle function.
    
    """
    
    # Evaluate circle function
    out = np.abs(r) <= 1
    
    return out.astype(int)  # Convert boolean array to integer (1/0)

def jinc(x):
    """
    Jinc function: J1(2*pi*x) / x
    
    """
    
    # Initialize output with pi (value for x=0)
    out = np.pi * np.ones_like(x)
    
    # Create a mask for non-zero elements of x
    mask = (x != 0)
    
    # Compute output values for all other x
    out[mask] = j1(2 * np.pi * x[mask]) / x[mask]
    
    return out

def rect(x):
    """
    Rectangle function.
    
    """
    
    # Evaluate rectangle function
    out = np.abs(x) <= 0.5
    
    return out.astype(int)  # Convert boolean array to integer (1/0)

def tri(x):
    """
    Triangle function.
    
    """
    
    # Create lines
    t = 1 - np.abs(x)
    
    # Keep lines for |x| <= 1, out = 0 otherwise
    mask = np.abs(x) <= 1
    out = np.where(mask, t, 0)  # Use np.where to apply the mask
    
    return out

def ucomb(x):
    """
    Unit sample 'comb' function.
    
    """
    
    # Round to the 10^-6 place
    x = np.round(x * 10**6) / 10**6
    
    # Create output array where out is 1 if x is an integer, else 0
    out = (np.remainder(x, 1) == 0).astype(int)  # Convert boolean to int (1 or 0)
    
    return out

def udelta(x):
    """
    Unit sample 'delta' function.
    
    """
    
    # Round to the 10^-6 place
    x = np.round(x * 10**6) / 10**6
    
    # Create output array where out is 1 if x == 0, else 0
    out = (x == 0).astype(int)  # Convert boolean to int (1 or 0)
    
    return out



def propTF(u1, L, lambda_, z):
    """
    Propagation - transfer function approach
    Assumes same x and y side lengths and uniform sampling
    
    Parameters:
    u1 - source plane field
    L - source and observation plane side length
    lambda_ - wavelength
    z - propagation distance
    
    Returns:
    u2 - observation plane field
    """
    M, N = u1.shape  # get input field array size
    dx = L / M  # sample interval
    k = 2 * np.pi / lambda_  # wavenumber
    
    fx = np.arange(-1/(2*dx), 1/(2*dx), 1/L)  # freq coords
    FX, FY = np.meshgrid(fx, fx)
    
    H = np.exp(-1j * np.pi * lambda_ * z * (FX**2 + FY**2))  # trans func
    H = fftshift(H)  # shift trans func
    U1 = fft2(fftshift(u1))  # shift, fft src field
    U2 = H * U1  # multiply
    u2 = ifftshift(ifft2(U2))  # inv fft, center obs field
    
    return u2


def propIR(u1, L, lambda_, z):
    """
    Propagation - impulse response approach
    Assumes same x and y side lengths and uniform sampling
    
    Parameters:
    u1 - source plane field
    L - source and observation plane side length
    lambda_ - wavelength
    z - propagation distance
    
    Returns:
    u2 - observation plane field
    """
    M, N = u1.shape  # get input field array size
    dx = L / M  # sample interval
    k = 2 * np.pi / lambda_  # wavenumber
    
    x = np.arange(-L/2, L/2, dx)  # spatial coords
    X, Y = np.meshgrid(x, x)
    
    h = 1/(1j*lambda_*z) * np.exp(1j*k/(2*z)*(X**2 + Y**2))  # impulse
    H = fft2(fftshift(h)) * dx**2  # create trans func
    U1 = fft2(fftshift(u1))  # shift, fft src field
    U2 = H * U1  # multiply
    u2 = ifftshift(ifft2(U2))  # inv fft, center obs field
    
    return u2



def propFF(u1, L1, lambda_, z):
    """
    Propagation - Fraunhofer pattern
    Assumes uniform sampling
    
    Parameters:
    u1 - source plane field
    L1 - source plane side length
    lambda_ - wavelength
    z - propagation distance
    
    Returns:
    u2 - observation plane field
    L2 - observation plane side length
    """
    M, N = u1.shape  # get input field array size
    dx1 = L1 / M  # source sample interval
    k = 2 * np.pi / lambda_  # wavenumber
    
    L2 = lambda_ * z / dx1  # obs sidelength
    dx2 = lambda_ * z / L1  # obs sample interval
    x2 = np.arange(-L2/2, L2/2, dx2)  # obs coords
    X2, Y2 = np.meshgrid(x2, x2)
    
    c = 1 / (1j * lambda_ * z) * np.exp(1j * k / (2 * z) * (X2**2 + Y2**2))
    u2 = c * ifftshift(fft2(fftshift(u1))) * dx1**2
    
    return u2, L2


def tilt(uin, L, lambda_, alpha, theta):
    """
    Tilt phasefront
    Uniform sampling assumed
    
    Parameters:
    uin - input field
    L - side length
    lambda_ - wavelength
    alpha - tilt angle
    theta - rotation angle (x axis 0)
    
    Returns:
    uout - output field
    """
    M, N = uin.shape  # get input field array size
    dx = L / M  # sample interval
    k = 2 * np.pi / lambda_  # wavenumber
    
    x = np.arange(-L/2, L/2, dx)  # coords
    X, Y = np.meshgrid(x, x)
    
    uout = uin * np.exp(1j * k * (X * np.cos(theta) + Y * np.sin(theta)) 
                        * np.tan(alpha))  # apply tilt
    
    return uout


def focus(uin, L, lambda_, zf):
    """
    Converging or diverging phase-front
    Uniform sampling assumed
    
    Parameters:
    uin - input field
    L - side length
    lambda_ - wavelength
    zf - focal distance (+ converge, - diverge)
    
    Returns:
    uout - output field
    """
    M, N = uin.shape  # get input field array size
    dx = L / M  # sample interval
    k = 2 * np.pi / lambda_  # wavenumber
    
    x = np.arange(-L/2, L/2, dx)  # coords
    X, Y = np.meshgrid(x, x)
    
    uout = uin * np.exp(-1j * k / (2 * zf) * (X**2 + Y**2))  # apply focus
    
    return uout


def seidel_5(u0, v0, X, Y, wd, w040, w131, w222, w220, w311):
    """
    Compute wavefront OPD for first 5 Seidel wavefront aberration coefficients + defocus

    Parameters:
    u0, v0 - normalized image plane coordinate
    X, Y - normalized pupil coordinate arrays (like from meshgrid)
    wd - defocus
    w040 - spherical
    w131 - coma
    w222 - astigmatism
    w220 - field curvature
    w311 - distortion

    Returns:
    w - wavefront OPD
    """
    beta = np.arctan2(v0, u0)  # image rotation angle
    u0r = np.sqrt(u0**2 + v0**2)  # image height

    # rotate grid
    Xr = X * np.cos(beta) + Y * np.sin(beta)
    Yr = -X * np.sin(beta) + Y * np.cos(beta)

    # Seidel polynomials
    rho2 = Xr**2 + Yr**2
    w = (wd * rho2 +
         w040 * rho2**2 +
         w131 * u0r * rho2 * Xr +
         w222 * u0r**2 * Xr**2 +
         w220 * u0r**2 * rho2 +
         w311 * u0r**3 * Xr)

    return w



def prop2step(u1, L1, L2, lambda_, z):
    """
    Propagation using the 2-step Fresnel diffraction method.
    
    Parameters:
    u1 : ndarray
        Complex field at the source plane.
    L1 : float
        Source plane side-length.
    L2 : float
        Observation plane side-length.
    lambda_ : float
        Wavelength.
    z : float
        Propagation distance.
        
    Returns:
    u2 : ndarray
        Output field at the observation plane.
    """
    
    M, N = u1.shape  # Input array size
    k = 2 * np.pi / lambda_  # Wavenumber

    # Source plane
    dx1 = L1 / M
    x1 = np.linspace(-L1/2, L1/2 - dx1, M)
    X, Y = np.meshgrid(x1, x1)
    
    # Apply quadratic phase factor for the first step
    u = u1 * np.exp(1j * k / (2 * z * L1) * (L1 - L2) * (X**2 + Y**2))
    
    # Fourier transform to frequency domain
    u = np.fft.fftshift(np.fft.fft2(u))

    # Dummy (frequency) plane
    fx1 = np.linspace(-1/(2*dx1), 1/(2*dx1) - 1/L1, M)
    FX1, FY1 = np.meshgrid(fx1, fx1)

    # Apply quadratic phase factor in frequency domain
    u *= np.exp(-1j * np.pi * lambda_ * z * L1 / L2 * (FX1**2 + FY1**2))
    
    # Inverse Fourier transform back to spatial domain
    u = np.fft.ifftshift(np.fft.ifft2(u))

    # Observation plane
    dx2 = L2 / M
    x2 = np.linspace(-L2/2, L2/2 - dx2, M)
    X, Y = np.meshgrid(x2, x2)

    # Apply quadratic phase factor for the second step and scale adjustment
    u2 = (L2 / L1) * u * np.exp(-1j * k / (2 * z * L2) * (L1 - L2) * (X**2 + Y**2))
    
    # Scale adjustment from x1 to x2
    u2 *= (dx1**2 / dx2**2)

    return u2

