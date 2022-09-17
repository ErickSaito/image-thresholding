import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def global_thresholding(image_path, limiar=125, show=False):
  img_name = image_path.split('/')[-1]
  img = cv.imread(image_path)
  gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

  gray_img[gray_img < limiar] = 0
  gray_img[gray_img >= limiar] = 255
  
  cv.imwrite(f'results/global_thresholding-{img_name}', gray_img)

  if show:
    plt.imshow(img, 'gray', vmin=0, vmax=255)
    plt.show()

def bernsen(image_path, cont_limit=15, neighborhood=5, show=False):
  np.seterr(over='ignore')
  img_name = image_path.split('/')[-1]
  img = cv.imread(image_path)
  gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
  copy_img = gray_img.copy()

  height = gray_img.shape[0]
  width = gray_img.shape[1]
  img_size = height * width
  radius = int(neighborhood/2)

  for i in range(radius + 1, height - radius):
    for j in range(radius + 1, width - radius):
      block = gray_img[i-radius:i+radius, j-radius:j+radius]

      z_minimum = block.min()
      z_maximum = block.max()
      threshold = (z_minimum + z_maximum)/2
      contrast = z_maximum - z_minimum

      if contrast < cont_limit:
        threshold_class = 255
      else:
        threshold_class = threshold

      if gray_img[i,j] < threshold_class:
        copy_img[i,j] = 0
      else:
        copy_img[i,j] = 255
        
  cv.imwrite(f'results/bernsen-{img_name}', copy_img)
  if show:
    plt.imshow(copy_img, 'gray', vmin=0, vmax=255)
    plt.show()

def niblack(image_path, k, neighborhood=15):
  img_name = image_path.split('/')[-1]
  img = cv.imread(image_path)
  gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
  copy_img = gray_img.copy()

  height = gray_img.shape[0]
  width = gray_img.shape[1]
  img_size = height * width
  radius = int(neighborhood/2)

  for i in range(radius + 1, height - radius):
    for j in range(radius + 1, width - radius):
      block = gray_img[i-radius:i+radius, j-radius:j+radius]

      mean = np.mean(block)
      std = np.std(block)

      threshold = mean + k * std

      if gray_img[i,j] < threshold:
        copy_img[i,j] = 255
      else:
        copy_img[i,j] = 0
        
  cv.imwrite(f'results/niblack-{img_name}', copy_img)
  if show:
    plt.imshow(copy_img, 'gray', vmin=0, vmax=255)
    plt.show()


# global_thresholding('images/baboon.pgm')
# bernsen('images/sonnet.pgm')
niblack('images/sonnet.pgm', 0.1, 25)