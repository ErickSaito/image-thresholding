import cv2 as cv
import numpy as np
import math
from image_manager import ImageManager
from matplotlib import pyplot as plt

class ImageThreshold():

  def __init__(self, image_path: str):
    self._manager = ImageManager(image_path)

  def global_thresholding(self, limiar=125, show=False):
    gray_img = self._manager.get_image()

    gray_img[gray_img < limiar] = 0
    gray_img[gray_img >= limiar] = 255
    
    self._manager.save_image('global_thresholding', gray_img)

    if show:
      plt.imshow(img, 'gray', vmin=0, vmax=255)
      plt.show()

  def bernsen(self, n=10, show=False):
    gray_img = self._manager.get_image()

    np.seterr(over='ignore')
    height, width, img_size, radius = self._manager.get_image_defitions(n)

    for i in range(radius + 1, height - radius):
      for j in range(radius + 1, width - radius):
        block = gray_img[i-radius:i+radius, j-radius:j+radius]

        threshold = int((block.min() + block.max())/2)

        if gray_img[i,j] < threshold:
          gray_img[i,j] = 0
        else:
          gray_img[i,j] = 255
          
    self._manager.save_image('bernsen', gray_img)

    if show:
      plt.imshow(gray_img, 'gray', vmin=0, vmax=255)
      plt.show()

  def niblack(self, k, n=15, show=False):
    gray_img = self._manager.get_image()

    height, width, img_size, radius = self._manager.get_image_defitions(n)

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
          
    self._manager.save_image('niblack', gray_img)

    if show:
      plt.imshow(gray_img, 'gray', vmin=0, vmax=255)
      plt.show()

  def sauvola_pietaksinen(self, k=0.5, n=9, r=128, show=False):
    gray_img = self._manager.get_image()

    height, width, img_size, radius = self._manager.get_image_defitions(n)

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

    self._manager.save_image('sauvola_pietaksinen', gray_img)

    if show:
      plt.imshow(gray_img, 'gray', vmin=0, vmax=255)
      plt.show()

  def phansalskar(self, n, k=0.25, r=0.5, p=2, q=10, show=False):
    gray_img = self._manager.get_image()
    height, width, img_size, radius = self._manager.get_image_defitions(n)

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
    
    self._manager.save_image('phansalskar', gray_img)

    if show:
      plt.imshow(gray_img, 'gray', vmin=0, vmax=255)
      plt.show()

  def contrast(self, n=10, show=False):
    gray_img = self._manager.get_image()
    height, width, img_size, radius = self._manager.get_image_defitions(n)

    for i in range(radius + 1, height - radius):
      for j in range(radius + 1, width - radius):
        block = gray_img[i-radius:i+radius, j-radius:j+radius]
        
        local_max = abs(block.max() - gray_img[i,j])
        local_min = abs(gray_img[i,j] - block.min())

        if (local_max > local_min):
          gray_img[i,j] = 255
        else:
          gray_img[i,j] = 0

    self._manager.save_image('contrast', gray_img)

    if show:
      plt.imshow(gray_img, 'gray', vmin=0, vmax=255)
      plt.show()

  def mean(self, n=10, show=False):
    gray_img = self._manager.get_image()
    height, width, img_size, radius = self._manager.get_image_defitions(n)

    for i in range(radius + 1, height - radius):
      for j in range(radius + 1, width - radius):
        block = gray_img[i-radius:i+radius, j-radius:j+radius]
        
        mean = np.mean(block)

        if (gray_img[i,j] > mean):
          gray_img[i,j] = 255
        else:
          gray_img[i,j] = 0

    self._manager.save_image('mean', gray_img)
    if show:
      plt.imshow(gray_img, 'gray', vmin=0, vmax=255)
      plt.show()

  def median(self, n=10, show=False):
    gray_img = self._manager.get_image()

    height, width, img_size, radius = self._manager.get_image_defitions( n)

    for i in range(radius + 1, height - radius):
      for j in range(radius + 1, width - radius):
        block = gray_img[i-radius:i+radius, j-radius:j+radius]
        
        median = np.median(block)

        if (gray_img[i,j] > median):
          gray_img[i,j] = 255
        else:
          gray_img[i,j] = 0


    self._manager.save_image('median', gray_img)

    if show:
      plt.imshow(gray_img, 'gray', vmin=0, vmax=255)
      plt.show()
