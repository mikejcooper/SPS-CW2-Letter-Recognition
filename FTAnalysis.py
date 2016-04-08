from __future__ import print_function

import numpy as np
from skimage import io
import matplotlib.pyplot as plt

# For viewing purposes:
def showGraph(data):
    plot = plt.figure()
    plot = plot.add_subplot( 111 )
    plot.axis('off')
    plot.imshow( np.array(data,dtype=int), cmap='gray' )

def fourierTransform(data):
    f = np.fft.fft2(data)                           # Fourier transform
    fShift = np.fft.fftshift(f)                     # Puts u=0,v=0 in the centre
    return fShift

def inverseFourierTransform(fourierData):
    fInverseShift = np.fft.ifftshift(fourierData)   # Puts u=0,v=0 in the centre
    fInverse = np.fft.ifft2(fInverseShift)          # Inverse fourier transform
    return fInverse

def magnitudeSpectrum(fourierData):
    return 20*np.log(np.abs(fourierData))          # Magnitude spectrum

def phaseSpectrum(fourierData):
    return np.angle(fourierData)                   # Phase spectrum

# High Pass Filter (HPF) - remove the low frequencies by masking with a rectangular window of size 60x60
def masking(data):
    rows, cols = data.shape
    data[rows/2-30:rows/2+30, cols/2-30:cols/2+30] = 0
    return data


# Read image
f = np.array(io.imread('characters/T7.GIF'), dtype=float)

# Manipulations
fourier = fourierTransform(f)
magSpec = magnitudeSpectrum(fourier)
inverseFourier = inverseFourierTransform(fourier)

showGraph(magSpec)
showGraph(inverseFourier)
plt.show()



