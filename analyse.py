# %%
from uncertainties import unumpy
from load_data import *
from plot_data import *
from background_threshold import *
from source_detection import *
from photometry import *
from logNm import *

# %%
# Exclude the vertical line and artefacts (e.g. edge effects) from our analysis of the background
cleanPixels = getCleanPixels()

# Calculate the background threshold
threshold = getBackgroundThreshold(cleanPixels).n
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
# image = image[0:1000, 0:1000]
image = image[::, ::]


# %%
maskedImage = image.filled(0)
plotZScale(maskedImage)
# %%

# Detect sources

sourceEllipses, apertureSums = detectSources(image)


# %%
# Photometry
# Add errors to the counts
aperSumsWithErr = unumpy.uarray(apertureSums, np.sqrt(apertureSums))

# Calculate magnitudes from the pixel sums
magnitudes = calcMagnitudes(apertureSums)



# %%

gradient, yintercept = fitLogNm(magnitudes, mCutoff=None)

# %%
print("List of sources identified")
print("--------------------------------")

for (ellipse, mag) in zip(sourceEllipses, magnitudes):
    x, y = ellipse[0]
    print("x: %.2f, y: %.2f, magnitude: %s" % (x, y, mag.format("%.2u")))
    
# %%

plotZScale(image.data)
plotEllipses(sourceEllipses)

# %%

plotZScale(maskedImage)
plotEllipses(sourceEllipses)

# %%

# sourceEllipses, aperSumsWithErr, magnitudes = loadCatalogue()
saveCatalogue(sourceEllipses, aperSumsWithErr, magnitudes)

# %%
