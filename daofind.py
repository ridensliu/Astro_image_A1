# %%

from astropy.stats import sigma_clipped_stats
from photutils.datasets import load_star_image
from photutils.detection import DAOStarFinder
from uncertainties import unumpy
from load_data import *
from plot_data import *
from background_threshold import *
from source_detection import *
from photometry import *
from logNm import *
from astropy.table import Table

# %%
# Exclude the vertical line and artefacts (e.g. edge effects) from our analysis of the background
cleanPixels = getCleanPixels()

# Calculate the background threshold
threshold = getBackgroundThreshold(cleanPixels).n
print(threshold)
# %%
# Load in the image and remove bad sections so we can detect sources
image = getImage()
image = maskBackground(image, threshold)
image = maskVerticalLine(image, 1410, 1470)
image = maskCircle(image, 1440, 3200, 300)


bleedingSources = [(2091, 2185, 3704, 3809),
                    (717, 867, 3195, 3423),
                    (917, 1034, 2695, 2843),
                    (851, 968, 2217, 2364)]


for (xmin, xmax, ymin, ymax) in bleedingSources:
    image = maskRectangle(image, xmin, xmax, ymin, ymax)


image = cropImage(image, ymin=500, ymax=4400, xmin=100, xmax=2350)
# %%

# Run on a small section of the image

# image = image[0:1000, 0:1000]
#image = image[0:1000, 0:1000]
image = image[::,::]
#%%
def daoFind(cleanimage, Nsigma):
    
    data = image.data
    
    mean, median, std = sigma_clipped_stats(data)  
    
    daofind = DAOStarFinder(fwhm=1.0, threshold=Nsigma * std)  
    
    sources = daofind(data - Nsigma * std)
    
    for col in sources.colnames:  
        
        sources[col].info.format = '%.8g'  
    
    # print(sources) 
    
    magn = sources['mag']

    gradient, yintercept = fitLogNm(magn, mCutoff=None)

# %%











