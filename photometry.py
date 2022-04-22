# %%
import numpy as np
from uncertainties import ufloat
from uncertainties import unumpy
from load_data import *
from plot_data import *
from ellipses import *


def getApertureSum(image, sourcePixels, ellipse, delta, plot=False):
    # Enlarge the ellipse to create an annular reference aperture
    widerEllipse = enlargeEllipse(ellipse, delta)
    i = 0

    # Get pixels within each ellipse
    widerEllipsePixels = getEllipsePixels(image, widerEllipse)
    annulusPixels = np.logical_xor(sourcePixels, widerEllipsePixels)


    # Find the sum of counts
    data = image.data
    bgCnt = np.sum(data[annulusPixels])
    sourceCnt = np.sum(data[sourcePixels])

    Nbg = np.count_nonzero(annulusPixels)
    Nsource = np.count_nonzero(sourcePixels)

    # print(bgCnt)
    # Subtract contribution from background
    sourceCnt -= Nsource * (bgCnt / Nbg)


    if sourceCnt < 0:
        # print("Warning: source at (%d, %d) has an aperture sum less than 0. Ignoring this sourcee." % (ellipse[0][0], ellipse[0][1]))
        # print(ellipse)
        # plot = True
        i+=1


    # Plot the result
    if plot:
        plotZScale(image.data, "gray")
        plotEllipses([ellipse, widerEllipse])
        x, y = ellipse[0]
        plt.title("source at (%.2f, %.2f)" % (x, y))

        # Zoom into this source
        boxSize = 50
        plt.xlim(x-boxSize, x+boxSize)
        plt.ylim(y-boxSize, y+boxSize)

        plt.show()


    return sourceCnt



def getApertureSumEllipse(image, ellipse, delta, plot=False):
    """
    Returns the sum of pixel counts within the given ellipse.

    Params
    -------
    image
    ellipse
    delta: The width of the annular reference aperture.
    plot: Whether to show a plot of the source and reference apertures.
    """

    # Enlarge the ellipse to create an annular reference aperture
    widerEllipse = enlargeEllipse(ellipse, delta)


    # Get pixels within each ellipse
    sourcePixels = getEllipsePixels(image, ellipse)
    widerEllipsePixels = getEllipsePixels(image, widerEllipse)
    annulusPixels = np.logical_xor(sourcePixels, widerEllipsePixels)

    # Find the sum of counts
    data = image.data
    bgCnt = np.sum(data[annulusPixels])
    sourceCnt = np.sum(data[sourcePixels])
    Nbg = np.count_nonzero(annulusPixels)
    Nsource = np.count_nonzero(sourcePixels)


    # Subtract contribution from background
    sourceCnt -= Nsource * (bgCnt / Nbg)


    if sourceCnt < 0:
        print("Source at (%d, %d) has an aperture sum less than 0." % (ellipse[0][0], ellipse[0][1]))
        print(ellipse)

        plot = True


    # Plot the result
    if plot:
        plotZScale(image.data, "gray")
        plotEllipses([ellipse, widerEllipse])
        x, y = ellipse[0]
        plt.title("source at (%.2f, %.2f)" % (x, y))

        # Zoom into this source
        boxSize = 100
        plt.xlim(x-boxSize, x+boxSize)
        plt.ylim(y-boxSize, y+boxSize)

        plt.show()
    
    

    return sourceCnt



def calcMagnitudes(aperSumsWithErr):
    zeroPt = ufloat(header["MAGZPT"], header["MAGZRR"])
    
    magnitudes = zeroPt - 2.5*unumpy.log10(aperSumsWithErr)

    return magnitudes

# %%
