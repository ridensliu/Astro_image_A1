# %%
import numpy as np
import matplotlib.pyplot as plt
from uncertainties import unumpy
from scipy.optimize import curve_fit
from load_data import *



def getCleanPixels():
    # Remove bad pixels
    # so they don't distort our analysis of the background
    image = getImage()
    image = maskVerticalLine(image, 1410, 1470)
    image = maskCircle(image, 1440, 3200, 300)
    image = cropImage(image, ymin=500, ymax=4400, xmin=100, xmax=2350)


    # Remove masked pixels from our analysis
    pixels = image.data[image.mask == False]


    return pixels



def getBackgroundThreshold(data, bins=50, Nsigma=5):
    """In order to eliminate the background,
we have to find a reasonable value for the background

Looking at the histogram of the background,
the distribution has a gaussian shape, therefore we fit a gaussian

The threshold is determined using 5 standard deviation from the mean

The function returns the threshold of the background"""

    data = data.flatten()

    # Find the region to restrict the pixel value histogram to
    # that just includes bg and excludes bright pixels (sources)
    lowBound, uppBound = getLowerAndUpperBackgroundBound(data)


    # Restrict to just pixels within the background range
    data = data[data >= lowBound]
    data = data[data <= uppBound]


    plt.figure(dpi=400)
    heights, bins, patches = plt.hist(data, bins, label="data")
    binCentres = (bins[1:] + bins[:-1])/2


    # Fit a Gaussian
    def gaussian(x, mu, sigma, A):
        return A*np.exp(-(x - mu)**2 / (2 * sigma**2))


    p0 = [np.mean(data), np.std(data), np.max(heights)]
    params, cov = curve_fit(gaussian, binCentres, heights, p0)
    errors = np.sqrt(np.diag(cov))

    # Make sure sigma is positive
    params[1] = np.abs(params[1])

    paramsWithErrors = unumpy.uarray(params, errors)
    mu, sigma, A = paramsWithErrors

    threshold = mu + Nsigma*sigma


    # Raise an error if we were unable to fit a Gaussian
    if (np.any(np.isinf(errors))):
        raise Exception("Unable to fit a Gaussian to the background.")



    # Plot the resulting fit
    x = np.linspace(binCentres[0], binCentres[-1], 500)
    plt.plot(x, gaussian(x, *params), label="Gaussian fit")
    plt.vlines(threshold.n, 0, heights.max(), color="r", linestyles="dotted", label="5$\sigma$ threshold")


    plt.xlabel("pixel value")
    ylabel = "number of pixels"
    plt.ylabel(ylabel)
    plt.legend()
    plt.title("histogram of pixel values")
    plt.show()


    # Print the results
    print("Gaussian fit parameters mu, sigma, A:")
    print([p.format("%.2u") for p in paramsWithErrors])
    print("")


    return threshold



def getLowerAndUpperBackgroundBound(data):
    data = data.flatten()
    heights, binEdges = np.histogram(data, bins=1000)
    bins = (binEdges[1:] + binEdges[:-1])/2

    maxIndex = np.argmax(heights)
    heightThreshold = heights[maxIndex]*0.01

    left = heights[:maxIndex]
    right = heights[maxIndex:]

    leftIndex = np.argmin(abs(left - heightThreshold))
    rightIndex = np.argmin(abs(right - heightThreshold))


    lowBound, uppBound = bins[leftIndex], bins[len(left) + rightIndex]


    return lowBound, uppBound


# %%
