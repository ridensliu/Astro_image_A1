import numpy as np
import pandas as pd
from uncertainties import unumpy
from astropy.io import fits


filePath = "A1_mosaic.fits"


def loadData(filePath):
    return fits.getdata(filePath)


def loadHeader(filePath):
    return fits.getheader(filePath, 0)


originalImage = loadData(filePath)
header = loadHeader(filePath)



def getImage():
    return np.ma.masked_array(originalImage, False, fill_value=0)



def maskBackground(image, threshold):
    data = image.data

    return np.ma.masked_array(data, data < threshold, fill_value=0)



def maskVerticalLine(image, xmin, xmax):
    height, width = image.shape

    for y in range(0, height):
        for x in range (xmin, xmax+1):
            image[y, x] = np.ma.masked

    return image



def maskCircle(image, x, y, radius):

    for dy in range(-radius, radius+1):
        for dx in range(-radius, radius+1):
            if (dx**2 + dy**2) < radius**2:
                image[y + dy, x + dx] = np.ma.masked

    return image



def maskRectangle(image, xmin, xmax, ymin, ymax):

    for y in range(ymin, ymax+1):
        for x in range(xmin, xmax+1):
            image[y, x] = np.ma.masked

    return image



def cropImage(image, xmin=0, xmax=None, ymin=0, ymax=None):
    return image[ymin:ymax, xmin:xmax]



def saveCatalogue(ellipses, apertureSums, magnitudes):
    xy, axLengths, angles = list(zip(*ellipses))
    x, y = list(zip(*xy))
    majorAxLengths, minorAxLengths = list(zip(*axLengths))

    aperSums = unumpy.nominal_values(apertureSums)
    aperSumsErrs = unumpy.std_devs(apertureSums)

    mags = unumpy.nominal_values(magnitudes)
    magsErrs = unumpy.std_devs(magnitudes)

    df = pd.DataFrame({"x": x, "y": y, "majoraxislength": majorAxLengths, "minoraxislength": minorAxLengths, "angle": angles, "count": aperSums, "counterr": aperSumsErrs, "magnitude": mags, "magnitudeerr": magsErrs})

    df.to_csv("catalogue.csv")



def loadCatalogue():
    df = pd.read_csv("catalogue.csv", index_col=0)


    x, y, majorAxLengths, minorAxLengths, angles, aperSums, aperSumsErrs, mags, magsErrs = df.to_numpy().T

    ellipses = []

    for i in range(len(x)):
        ellipse = ((x[i], y[i]), (majorAxLengths[i], minorAxLengths[i]), angles[i])
        ellipses.append(ellipse)

    aperSumsWithErrs = unumpy.uarray(aperSums, aperSumsErrs)
    magnitudes = unumpy.uarray(mags, magsErrs)


    return ellipses, aperSumsWithErrs, magnitudes


# %%
