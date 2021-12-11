import sys
import skimage.io
import skimage.viewer
import skimage.draw

import pandas as pd
import matplotlib.pyplot as plt


# read original image, in full color, based on command
# line argument
image = skimage.io.imread(fname=sys.argv[1])

# display the original image
viewer = skimage.viewer.ImageViewer(image)
viewer.show()

# create a circular mask to select the 7th well in the first row
# WRITE YOUR CODE HERE
mask = np.zeros(shape=image.shape[0:2], dtype="bool")
circle = skimage.draw.circle(240, 1053, radius=49, shape=image.shape[:2])
mask[circle] = 1

# just for display:
# make a copy of the image, call it masked_image, and
# use np.logical_not() and indexing to apply the mask to it
# WRITE YOUR CODE HERE
masked_img = image[:]
masked_img[np.logical_not(mask)] = 0

# create a new window and display maskedImg, to verify the
# validity of your mask
# WRITE YOUR CODE HERE
viewer = skimage.viewer.ImageViewer(masked_img)
viewer.show()

# list to select colors of each channel line
colors = ("r", "g", "b")
channel_ids = (0, 1, 2)

# create the histogram plot, with three lines, one for
# each color
plt.xlim([0, 256])
for (channel_id, c) in zip(channel_ids, colors):
    # change this to use your circular mask to apply the histogram
    # operation to the 7th well of the first row
    # MODIFY CODE HERE
    histogram, bin_edges = np.histogram(
        image[:, :, channel_id][mask], bins=256, range=(0, 256)
    )

    plt.plot(histogram, color=c)

plt.xlabel("color value")
plt.ylabel("pixel count")

plt.show()