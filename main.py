import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def open_image(image_path: str):
  img = cv.imread(image_path)
  gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

  return gray_img

def save_image(image_path: str, name: str, img):
  img_name = image_path.split('/')[-1]
  cv.imwrite(f'results/{name}-{img_name}', img)

def get_image_defitions(img, n):
  height = img.shape[0]
  width = img.shape[1]
  img_size = height * width
  radius = int(n/2)

  return height, width, img_size, radius

def global_thresholding(image_path, limiar=125, show=False):
  gray_img = open_image(image_path)

  gray_img[gray_img < limiar] = 0
  gray_img[gray_img >= limiar] = 255
  
  save_image(image_path, 'global_thresholding', gray_img)

  if show:
    plt.imshow(img, 'gray', vmin=0, vmax=255)
    plt.show()

def bernsen(image_path, cont_limit=15, neighborhood=5, show=False):
  np.seterr(over='ignore')
  gray_img = open_image(image_path)
  copy_img = gray_img.copy()

  height, width, img_size, radius = get_image_defitions(gray_img, neighborhood)

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
        
  save_image(image_path, 'bernsen', copy_img)
  if show:
    plt.imshow(copy_img, 'gray', vmin=0, vmax=255)
    plt.show()

def niblack(image_path, k, neighborhood=15, show=False):
  gray_img = open_image(image_path)
  copy_img = gray_img.copy()

  height, width, img_size, radius = get_image_defitions(gray_img, neighborhood)

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
        
  save_image(image_path, 'niblack', copy_img)

  if show:
    plt.imshow(copy_img, 'gray', vmin=0, vmax=255)
    plt.show()


# global_thresholding('images/baboon.pgm')
# bernsen('images/sonnet.pgm')
niblack('images/sonnet.pgm', 0.1, 25)