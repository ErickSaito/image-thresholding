import cv2 as cv

class ImageManager():

  def __init__(self, image_path: str):
    self._image_path = image_path
    self._image = cv.cvtColor(cv.imread(self._image_path), cv.COLOR_BGR2GRAY)

  def get_image(self):
    return self._image

  def save_image(self, process_name: str, img):
    image_name = self._image_path.split('/')[-1].split('.')[0]
    cv.imwrite(f'results/{image_name}-{process_name}.png', img)

  def get_image_defitions(self, n):
    height, width = self._image.shape
    img_size = self._image.size
    radius = int(n/2)

    return height, width, img_size, radius
