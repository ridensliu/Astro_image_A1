# %%
import cv2
from skimage.segmentation import flood
import numpy as np
from uncertainties import unumpy
from plot_data import *
from load_data import *
from ellipses import *
from photometry import *



def detectSources(image, N=-1, debug=False):
    height, width = image.shape
    # Pixels in order of brightest to dimmest
    pixelIndices = np.flip(np.argsort(image.filled(0), axis=None))


    sourceEllipses = []
    apertureSums = []


    for index in pixelIndices:
        if N != -1 and len(sourceEllipses) >= N:
            break


        # This is quicker than using unravel_index
        y, x = index % height, index // height



        # Check whether this pixel is masked
        if (image[y, x] is np.ma.masked):
            continue


        # Check whether this pixel lies near an edge
        # If so, skip this pixel to avoid errors from edge effects
        edgeThreshold = 50

        if (x < edgeThreshold) or (x > width - edgeThreshold) or (y < edgeThreshold) or (y > height - edgeThreshold):
            if debug:
                print("Skipping source at (%d, %d) as it is too close to image edge." % (x, y))
            continue



        # Fit an ellipse to the source and store it
        cleanData = getPixelsWithinSource(image, x, y)
        success, ellipse = fitEllipseToCleanData(cleanData, x, y)

        if not success:
            continue


        # Perform checks on the ellipse
        isValidEllipse, shouldMask = checkValidEllipse(x, y, ellipse, debug)

        if not isValidEllipse:
            if shouldMask:
                # Mask the pixels connected to this source
                image.mask = np.logical_or(image.mask, cleanData)

            continue


        if debug:
            print(ellipse)


        # Enlarge the ellipse, to give some "buffer room" and include pixels just outside the ellipse that may be part of the source
        ellipse = enlargeEllipse(ellipse, 3)
        try:
            ellipsePixels = getEllipsePixels(image, ellipse)

        except Exception as e:
            # Sometimes this returns an error
            # so wrap it in a try/except block

            print("An error occured with the ellipse at (%d, %d):" % (x, y))
            print(e)

            continue

        sourceCnt = getApertureSum(image, ellipsePixels, ellipse, delta=25)


        if sourceCnt > 0:
            sourceEllipses.append(ellipse)
            apertureSums.append(sourceCnt)


        # Mask the region contained within the ellipse
        image.mask = np.logical_or(image.mask, ellipsePixels)


    apertureSums = np.array(apertureSums)

    return sourceEllipses, apertureSums



def getPixelsWithinSource(image, x, y):

    # Turn the image into a series of 0s and 1s
    # based on whether pixels are source or bg/masked
    mask = image.mask
    cleanData = np.logical_not(mask).astype(np.uint8)



    # Find the group of pixels connected to the source using skimage.segmentation.flood

    # Consider pixels to be connected if they're connected vertically or horizontally
    footprint = [[0,1,0],
                [1,1,1],
                [0,1,0]]


    # Beware: position is (y, x), not (x, y)!
    cleanData = flood(cleanData, (y, x), selem=footprint).astype(np.uint8)


    return cleanData



def fitEllipseToCleanData(cleanData, x, y):
    # Find contours

    # cv2.RETR_EXTERNAL to find the outermost contour

    contours, hierarchy = cv2.findContours(cleanData, cv2.RETR_EXTERNAL,  cv2.CHAIN_APPROX_SIMPLE)


    # plt.figure(dpi=400)
    # plt.imshow(cv2.drawContours(np.zeros(data.shape), contours, -1, (255, 255, 255), 2))
    # plt.show()


    if (len(contours) > 1):
        raise Exception("More than one contour was found for source at (%d, %d)." % (x, y))


    # Fit an ellipse
    cnt = contours[0]

    if len(cnt) < 5:
        # print("Need 5 points to fit an ellipse to the source at (%d, %d)." % (x, y))

        # (success, ellipse)
        return (False, None)


    ellipse = cv2.fitEllipse(cnt)

    # (success, ellipse)
    return (True, ellipse)



def checkValidEllipse(x, y, ellipse, debug):
    isValidEllipse = True
    shouldMask = False


    (xEl, yEl), (majorAxLength, minorAxLength), angle = ellipse


    # Check position of ellipse centre
    centreDistThreshold = 15
    if (x-xEl)**2 + (y-yEl)**2 > centreDistThreshold**2:
        if debug:
            print("Warning: source at (%d, %d) produced a ellipse aligned off centre. Ignoring this source." % (x, y))

        isValidEllipse = False


    # Check size of ellipse
    lengthThreshold = 150
    if (majorAxLength > lengthThreshold) or (minorAxLength > lengthThreshold):
        if debug:
            print("Warning: source at (%d, %d) produced a very large ellipse. Ignoring this source." % (x, y))

        isValidEllipse = False
        shouldMask = True


    return isValidEllipse, shouldMask


# %%