from os import path
import time
import math
import sys
from PIL import Image
from numpy import array
import numpy as np

np.set_printoptions(threshold=100000)


def kernelConvolution(kernelH, kernelV, pixels):
    edge = np.zeros((pixels.shape[0], pixels.shape[1]))
    for i in range(1, pixels.shape[0] - 1):
        for j in range(1, pixels.shape[1] - 1):
            center = pixels[i-1:i+2,j-1:j+2]
            h = abs(np.sum(kernelH * center))
            v = abs(np.sum(kernelV * center))
            #edge[i][j] = min(math.sqrt(h * h + v * v) * 1, 260)
            #edge[i][j] = math.sqrt(h * h + v * v) * 0.8
            edge[i][j] = min(math.sqrt(h * h + v * v), 2000)
    return edge



# Prompt user for file.
imageFile = input(" >> Image File Path: ")
while not(path.exists(imageFile)):
    print("  Not a valid file path")
    imageFile = input(" >> Image File Path: ")

# Set timestamp.
start = time.time()

# Convert image to array.
img = Image.open(imageFile)
rgb = np.array(img.convert(mode='RGB'))


# Kernels
kh = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
kv = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])

# Convolve each color plane.          
rgb[:,:,0] = kernelConvolution(kh, kv, rgb[:,:,0])
rgb[:,:,1] = kernelConvolution(kh, kv, rgb[:,:,1])
rgb[:,:,2] = kernelConvolution(kh, kv, rgb[:,:,2])



#rgb = np.multiply(rgb, 0.8, casting='unsafe').astype(np.uint8)
#rgb = np.multiply(rgb, 240 / np.amax(rgb), casting='unsafe').astype(np.uint8)
#rgb = rgb.astype(np.uint8)
#print(rgb)
#print(rgb.astype(np.uint8))
#print(rgb)
#print(np.amax(rgb))
#rgb = rgb * (240/np.amax(rgb))
#print(rgb)
#rgb = rgb.astype(np.uint8)
#print(rgb)
# Convert array to image and show.


m = np.amax(rgb)
for i in range(rgb.shape[0]):
    for j in range(rgb.shape[1]):
        for k in range(rgb.shape[2]):
            rgb[i][j][k] = rgb[i][j][k] * 250 / m

edges = Image.fromarray(rgb)
edges.show()
edges.save('result/{}'.format(imageFile))



# Print time.
print(' {} seconds'.format(round(time.time() - start), 2))