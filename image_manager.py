import cv2 as cv

class ImageManager():

  def __init__(self, image_path: str):
    self._image_path = image_path
    self.image = cv.cvtColor(cv.imread(self._image_path), cv.COLOR_BGR2GRAY)

  def save_image(self, process_name: str):
    image_name = self._image_path.split('/')[-1]
    cv.imwrite(f'results/{process_name}-{image_name}', self.image)

  def get_image_defitions(self, n):
    height, width = self.image.shape
    img_size = self.image.size
    radius = int(n/2)

    return height, width, img_size, radius
