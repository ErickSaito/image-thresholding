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

global_thresholding('images/baboon.pgm')