import cv2 as cv
import numpy as np
import math
from image_manager import ImageManager
from matplotlib import pyplot as plt

def global_thresholding(image_path, limiar=125, show=False):
  manager = ImageManager(image_path)
  gray_img = manager.image

  gray_img[gray_img < limiar] = 0
  gray_img[gray_img >= limiar] = 255
  
  manager.image = gray_img
  manager.save_image('global_thresholding')

  if show:
    plt.imshow(img, 'gray', vmin=0, vmax=255)
    plt.show()

def bernsen(image_path, n=10, show=False):
  manager = ImageManager(image_path)
  gray_img = manager.image

  np.seterr(over='ignore')
  height, width, img_size, radius = manager.get_image_defitions(n)

  for i in range(radius + 1, height - radius):
    for j in range(radius + 1, width - radius):
      block = gray_img[i-radius:i+radius, j-radius:j+radius]

      threshold = int((block.min() + block.max())/2)

      if gray_img[i,j] < threshold:
        gray_img[i,j] = 0
      else:
        gray_img[i,j] = 255
        
  manager.image = gray_img
  manager.save_image('bernsen')

  if show:
    plt.imshow(gray_img, 'gray', vmin=0, vmax=255)
    plt.show()

def niblack(image_path, k, n=15, show=False):
  manager = ImageManager(image_path)
  gray_img = manager.image

  height, width, img_size, radius = manager.get_image_defitions(n)

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
        
  manager.image = gray_img
  manager.save_image('niblack')

  if show:
    plt.imshow(gray_img, 'gray', vmin=0, vmax=255)
    plt.show()

def sauvola_pietaksinen(image_path, k=0.5, n=7, r=128, show=False):
  manager = ImageManager(image_path)
  gray_img = manager.image

  height, width, img_size, radius = manager.get_image_defitions(n)

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

  manager.image = gray_img
  manager.save_image('sauvola_pietaksinen')

  if show:
    plt.imshow(gray_img, 'gray', vmin=0, vmax=255)
    plt.show()

def phansalskar(image_path, n, k=0.25, r=0.5, p=2, q=10, show=False):
  manager = ImageManager(image_path)
  gray_img = manager.image
  height, width, img_size, radius = manager.get_image_defitions(n)

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
  
  manager.image = gray_img
  manager.save_image('phansalskar')

  if show:
    plt.imshow(gray_img, 'gray', vmin=0, vmax=255)
    plt.show()

def contrast(image_path, n=10, show=False):
  manager = ImageManager(image_path)
  gray_img = manager.image

  height, width, img_size, radius = manager.get_image_defitions(n)

  for i in range(radius + 1, height - radius):
    for j in range(radius + 1, width - radius):
      block = gray_img[i-radius:i+radius, j-radius:j+radius]
      
      local_max = abs(block.max() - gray_img[i,j])
      local_min = abs(gray_img[i,j] - block.min())

      if (local_max > local_min):
        gray_img[i,j] = 255
      else:
        gray_img[i,j] = 0

  manager.image = gray_img
  manager.save_image('contrast')

  if show:
    plt.imshow(gray_img, 'gray', vmin=0, vmax=255)
    plt.show()

def mean(image_path, n=10, show=False):
  manager = ImageManager(image_path)
  gray_img = manager.image

  height, width, img_size, radius = manager.get_image_defitions(n)

  for i in range(radius + 1, height - radius):
    for j in range(radius + 1, width - radius):
      block = gray_img[i-radius:i+radius, j-radius:j+radius]
      
      mean = np.mean(block)

      if (gray_img[i,j] > mean):
        gray_img[i,j] = 255
      else:
        gray_img[i,j] = 0

  manager.image = gray_img
  manager.save_image('mean')
  if show:
    plt.imshow(gray_img, 'gray', vmin=0, vmax=255)
    plt.show()

def median(image_path, n=10, show=False):
  manager = ImageManager(image_path)
  gray_img = manager.image

  height, width, img_size, radius = manager.get_image_defitions( n)

  for i in range(radius + 1, height - radius):
    for j in range(radius + 1, width - radius):
      block = gray_img[i-radius:i+radius, j-radius:j+radius]
      
      median = np.median(block)

      if (gray_img[i,j] > median):
        gray_img[i,j] = 255
      else:
        gray_img[i,j] = 0


  manager.image = gray_img
  manager.save_image('median')

  if show:
    plt.imshow(gray_img, 'gray', vmin=0, vmax=255)
    plt.show()

global_thresholding('images/monarch.pgm')
bernsen('images/retina.pgm')
niblack('images/monarch.pgm', 1, 7)
sauvola_pietaksinen('images/monarch.pgm')
phansalskar('images/retina.pgm', 5)
contrast('images/retina.pgm')
mean('images/monarch.pgm')
median('images/sonnet.pgm')