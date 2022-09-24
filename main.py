import cv2 as cv
import numpy as np
import math
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
  img_size = img.size
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

def bernsen(image_path, n=10, show=True):
  np.seterr(over='ignore')
  gray_img = open_image(image_path)

  height, width, img_size, radius = get_image_defitions(gray_img, n)

  for i in range(radius + 1, height - radius):
    for j in range(radius + 1, width - radius):
      block = gray_img[i-radius:i+radius, j-radius:j+radius]

      threshold = int((block.min() + block.max())/2)

      if gray_img[i,j] < threshold:
        gray_img[i,j] = 0
      else:
        gray_img[i,j] = 255
        
  save_image(image_path, 'bernsen', gray_img)
  if show:
    plt.imshow(gray_img, 'gray', vmin=0, vmax=255)
    plt.show()

def niblack(image_path, k, n=15, show=False):
  gray_img = open_image(image_path)

  height, width, img_size, radius = get_image_defitions(gray_img, n)

  for i in range(radius + 1, height - radius):
    for j in range(radius + 1, width - radius):
      block = gray_img[i-radius:i+radius, j-radius:j+radius]

      mean = np.mean(block)
      std = np.std(block)

      threshold = mean + k * std

      if gray_img[i,j] < threshold:
        gray_img[i,j] = 0
      else:
        gray_img[i,j] = 255
        
  save_image(image_path, 'niblack', gray_img)

  if show:
    plt.imshow(gray_img, 'gray', vmin=0, vmax=255)
    plt.show()

def sauvola_pietaksinen(image_path, k=0.5, n=7, r=128, show=False):
  gray_img = open_image(image_path)

  height, width, img_size, radius = get_image_defitions(gray_img, n)

  for i in range(radius + 1, height - radius):
    for j in range(radius + 1, width - radius):
      block = gray_img[i-radius:i+radius, j-radius:j+radius]

      mean = np.mean(block)
      std = np.std(block)

      threshold = mean * (1 + k * ((std / r) - 1))

      if gray_img[i,j] < threshold:
        gray_img[i,j] = 0
      else:
        gray_img[i,j] = 255

  save_image(image_path, 'sauvola_pietaksinen', gray_img)

  if show:
    plt.imshow(gray_img, 'gray', vmin=0, vmax=255)
    plt.show()

def phansalskar(image_path, n, k=0.25, r=0.5, p=2, q=10, show=False):
  gray_img = open_image(image_path)
  height, width, img_size, radius = get_image_defitions(gray_img, n)

  r = r * 256
  for i in range(radius + 1, height - radius):
    for j in range(radius + 1, width - radius):
      block = gray_img[i-radius:i+radius, j-radius:j+radius]

      mean = np.mean(block)
      std = np.std(block)

      threshold = mean * (1 + p * math.exp((q * (-1)) * mean) + (k * ((std/r) - 1)))

      if gray_img[i,j] < threshold:
        gray_img[i,j] = 0
      else:
        gray_img[i,j] = 255
  
  save_image(image_path, 'phansalskar', gray_img)

  if show:
    plt.imshow(gray_img, 'gray', vmin=0, vmax=255)
    plt.show()


# global_thresholding('images/baboon.pgm')
# bernsen('images/baboon.pgm')
# niblack('images/monarch.pgm', 1, 7)
# sauvola_pietaksinen('images/monarch.pgm')
# phansalskar('images/retina.pgm', 5)