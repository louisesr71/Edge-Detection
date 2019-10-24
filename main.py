from os import path
import time
import math
import sys
from PIL import Image
from numpy import array
import numpy as np

np.set_printoptions(threshold=100000)


#time1, time2 = 0, 0

def kernelConvolution(kernelH, kernelV, pixels):
    edge = np.zeros((pixels.shape[0], pixels.shape[1]))
    for i in range(1, pixels.shape[0] - 1):
        for j in range(1, pixels.shape[1] - 1):
            #global time1
            #global time2
            #since1 = time.time()
            center = pixels[i-1:i+2,j-1:j+2]
            #time1 += time.time() - since1
            #since2 = time.time()
            h = abs(np.sum(kernelH * center))
            v = abs(np.sum(kernelV * center))
            #time2 += time.time() - since2
            edge[i][j] = math.sqrt(h * h + v * v)
    return edge


scale = sys.argv[1] == '-scale'
if scale:
    imageFile = sys.argv[2]
else:
    imageFile = sys.argv[1]

# Set timestamp.
start = time.time()

# Convert image to array.
img = Image.open(imageFile)
rgb = np.array(img.convert(mode='RGB'))


# Kernels
kh = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
kv = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])

# Convolve each color plane.          
#rgb[:,:,0] = kernelConvolution(kh, kv, rgb[:,:,0])
#rgb[:,:,1] = kernelConvolution(kh, kv, rgb[:,:,1])
#rgb[:,:,2] = kernelConvolution(kh, kv, rgb[:,:,2])

red = kernelConvolution(kh, kv, rgb[:,:,0])
green = kernelConvolution(kh, kv, rgb[:,:,1])
blue = kernelConvolution(kh, kv, rgb[:,:,2])

m = max(np.amax(red), np.amax(green), np.amax(blue))
sf = 250 / m

if scale:
    # Cubic scaling.
    red = ((red * sf - 125) ** 3) / 125**2 + 125
    green = ((green * sf - 125) ** 3) / 125**2 + 125
    blue = ((blue * sf - 125) ** 3) / 125**2 + 125
else:
    # No scaling.
    red *= sf
    green *= sf
    blue *= sf

#color = np.hstack((red, green, blue)) # 3 images in a row
color = np.dstack((red, green, blue))
edges = Image.fromarray(color.astype(np.uint8), mode='RGB')
edges.show()
edges.save('result/{}'.format(imageFile))

#print(time1)
#print(time2)

# Print time.
print(' {} seconds'.format(round(time.time() - start), 2))