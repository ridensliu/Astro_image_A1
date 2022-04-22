import cv2
import numpy as np


def getEllipsePixels(image, ellipse):
    
    ellipsePixels = cv2.ellipse(np.zeros(image.shape), ellipse, (255,255,255), cv2.FILLED).astype(bool)

    return ellipsePixels


def enlargeEllipse(ellipse, delta):
    # Ellipse format given by opencv
    (x, y), (majorAxLength, minorAxLength), angle = ellipse

    return ((x, y), (majorAxLength + delta, minorAxLength + delta), angle)
