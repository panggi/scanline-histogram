import numpy as np
import matplotlib.pyplot as plt
import cv2
# import matplotlib.image as mpimg
# from numpy import *
from PIL import Image

# def rgb2gray(rgb):
#       return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

# def histeq(im,nbr_bins=256):
#
#    #get image histogram
#    imhist,bins = histogram(im.flatten(),nbr_bins,normed=True)
#    cdf = imhist.cumsum() #cumulative distribution function
#    cdf = 255 * cdf / cdf[-1] #normalize
#
#    #use linear interpolation of cdf to find new pixel values
#    im2 = interp(im.flatten(),bins[:-1],cdf)
#
#    return im2.reshape(im.shape), cdf
def get_num_pixels(filepath):
    width, height = Image.open(open(filepath)).size
    return width,height

def chunks(l, n):
    n = max(1, n)
    return [l[i:i + n] for i in range(0, len(l), n)]

# img = mpimg.imread('dawkins-200px.png')
# gray = rgb2gray(img)
# equalized,cdf = histeq(gray)
# plt.imshow(equalized, cmap = plt.get_cmap('gray'))
# plt.show()
# img = Image.open("dawkins-200px.png").convert('LA')
# img.save("dawkins-gray.png")
# img_gray = Image.open("dawkins-gray.png")
# img_gray = array(Image.open('dawkins-gray.png').convert('L'))
# img_histeq, cdf = histeq(img_gray)
# img_histeq.save("dawkins-equalized.png")
# print 'width: %d - height: %d' % img_gray.size
img_to_equalize = cv2.imread('images/dawkins-200px.png',0)
equ = cv2.equalizeHist(img_to_equalize)
# res = np.hstack((img_to_equalize,equ)) #stacking images side-by-side
cv2.imwrite('images/dawkins-equalized.png',equ)

width,height = get_num_pixels('images/dawkins-equalized.png')
# print "Width: ", width
# print "Height: ", height
# print "Total pixel: ", width * height

equalized = Image.open('images/dawkins-equalized.png', 'r')
width, height = equalized.size
pixel_values = list(equalized.getdata())

scanline = []
for group in chunks(pixel_values, 200):
    scanline.append(group)

pix_loc = list(chunks(range(1,(width*height)), width))

print "Pixel location: "
print pix_loc[0] #first scanline
print "Scanline: "
print scanline[0] #first scanline

