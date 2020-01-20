from os import path
import time
import math
import sys
from PIL import Image
from numpy import array
import numpy as np

np.set_printoptions(threshold=100000)


#time1, time2 = 0, 0

def blur(pixels):
    img = np.zeros((pixels.shape[0], pixels.shape[1]))
    kernel = np.array([[1,1,1],[1,1,1],[1,1,1]])
    for i in range(1, pixels.shape[0] - 1):
        for j in range(1, pixels.shape[1] - 1):
            center = pixels[i-1:i+2,j-1:j+2]
            img[i][j] = np.sum(kernel * center) / 9.0
    return img


def kernelConvolution(kernelH, kernelV, pixels):
    edge = np.zeros((pixels.shape[0], pixels.shape[1]))
    for i in range(1, pixels.shape[0] - 1):
        for j in range(1, pixels.shape[1] - 1):
            center = pixels[i-1:i+2,j-1:j+2]
            h = abs(np.sum(kernelH * center))
            v = abs(np.sum(kernelV * center))
            edge[i][j] = math.sqrt(h * h + v * v)
    return edge


def detect_edges(pixels):
    edge = np.zeros((pixels.shape[0], pixels.shape[1]))
    x = get_x_shift(pixels) - pixels
    y = get_y_shift(pixels) - pixels
    for i in range(1, pixels.shape[0] - 1):
        for j in range(1, pixels.shape[1] - 1):
            h = 1 * x[i-1][j+1] + 2 * x[i][j+1] + 1 * x[i+1][j+1]
            v = 1 * y[i+1][j-1] + 2 * y[i+1][j] + 1 * y[i+1][j+1]
            edge[i][j] = np.sqrt(h * h + v * v)
    return edge
            

def get_x_shift(pixels):
    transformed = np.copy(pixels)
    for i in range(transformed.shape[0]):
        transformed[i] = np.append([0, 0], transformed[i][:-2])
    return transformed

def get_y_shift(pixels):
    transformed = np.copy(pixels)
    transformed = np.append(np.zeros((2, transformed.shape[1])), transformed[:-2], axis=0)
    return transformed

scale = '-scale' in sys.argv
blurred = '-blur' in sys.argv
imageFile = sys.argv[len(sys.argv) - 1]

# Set timestamp.
start = time.time()

# Convert image to array.
img = Image.open(imageFile)
rgb = np.array(img.convert(mode='RGB')).astype(float)


# Kernels
kh = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
kv = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])

# Separate color planes.
red = rgb[:,:,0]
green = rgb[:,:,1]
blue = rgb[:,:,2]

if blurred:
    blur(red)
    blur(green)
    blur(blue)


#red = kernelConvolution(kh, kv, red)
#green = kernelConvolution(kh, kv, green)
#blue = kernelConvolution(kh, kv, blue)
red = detect_edges(red)
green = detect_edges(green)
blue = detect_edges(blue)

m = max(np.amax(red), np.amax(green), np.amax(blue))
sf = 250 / m

if scale:
    # Cubic scaling:
    #red = ((red * sf - 125) ** 3) / 125**2 + 125
    #green = ((green * sf - 125) ** 3) / 125**2 + 125
    #blue = ((blue * sf - 125) ** 3) / 125**2 + 125
    
    # Modified cubic scaling:
    red = (1.0/25625.0)*(red * sf - 125)**3 + (10000.0/25625.0)*(red * sf - 125) + 125
    green = (1.0/25625.0)*(green * sf - 125)**3 + (10000.0/25625.0)*(green * sf - 125) + 125
    blue = (1.0/25625.0)*(blue * sf - 125)**3 + (10000.0/25625.0)*(blue * sf - 125) + 125
    
    # Signmoid scaling:
    #red = 250 / (1 + math.e**(-(red * sf - 125)/25))
    #green = 250 / (1 + math.e**(-(green * sf - 125)/25))
    #blue = 250 / (1 + math.e**(-(blue * sf - 125)/25))
    
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