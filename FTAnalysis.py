from __future__ import print_function

import os
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sg


# For viewing purposes:
def showGraph(data):
    plot = plt.figure()
    plot = plot.add_subplot( 111 )
    plot.axis('off')
    plot.imshow(np.array(data,dtype=int), cmap='gray')

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
    return np.unwrap(np.angle(fourierData))                # Phase spectrum

# High Pass Filter (HPF) - remove the low frequencies by masking with a rectangular window of size min(rows,cols)/6
def masking(data):
    rows, cols = data.shape
    mf = min(rows,cols)/6   # Masking Factor
    data[rows/2-mf:rows/2+mf, cols/2-mf:cols/2+mf] = 1
    return data

def convolutionFilter(image,vORh):
    if vORh == 'v':
        # return sg.convolve(image, [[1,-1]], "valid")         # Vertical Data
        return sg.fftconvolve(image, [[1,-1]], "valid")         # Vertical Data
    else:
        # return sg.convolve(image, [[1],[-1]], "valid")       # Horizontal Data
        return sg.fftconvolve(image, [[1],[-1]], "valid")       # Horizontal Data

def commonSize(a1, a2):
    # Chop-off the first rows and cols from the two numpy arrays a1
    # and a2 so that they end up having the same size.
    (r1, c1) = a1.shape
    (r2, c2) = a2.shape
    return (a1[r1-r2 if r1>r2 else 0:,
               c1-c2 if c1>c2 else 0:],
            a2[r2-r1 if r2>r1 else 0::,
               c2-c1 if c2>c1 else 0:])

def convolutionCombination(image):
    imv, imh = commonSize(convolutionFilter(image,'h'),convolutionFilter(image,'v'))
    return np.sqrt(np.power(imv, 2)+np.power(imh, 2))

def getAllImages(letter):
    images = []
    dir = 'characters/'
    for subdir, dirs, files in os.walk(dir):
        for file in files:
            if os.path.isfile(os.path.join(dir,file)) and letter in file:
                    images.append(np.array(io.imread(dir + file), dtype=float))
    return images

def getAllMagSpec(images):
    result = []
    for i in images:
        result.append(magnitudeSpectrum(convolutionCombination(fourierTransform(i))))
    return result



def main():
    # Read image


    imageRGB = io.imread('characters/V5.GIF')
    image = np.array(imageRGB, dtype=float)
    image = convolutionCombination(image)
    # Manipulations
    fourier = fourierTransform(image)

    magSpec = magnitudeSpectrum(fourier)
    phaseSpec = phaseSpectrum(fourier)




    inverseFourier = inverseFourierTransform(fourier)

    # images = readInImages('T')


    showGraph(magSpec)
    # showGraph(phaseSpec)
    # showGraph(inverseFourier)
    plt.show()




if __name__ == "__main__":
    main()

