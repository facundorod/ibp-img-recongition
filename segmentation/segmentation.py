import cv2
import numpy as np
class SegmentationService():
  def __init__(self, image): 
    self.image = image

  def start_process(self):
    self.__convert_to_hsv()
    self.__thresold_segmentation()
    self.__convert_to_gray()
    self.__convert_to_binary()
    self.image = cv2.GaussianBlur(self.image, (5, 5), 0)
    self.__closing(iterations=2)
    

  def __convert_to_binary(self):
    _, self.image = cv2.threshold(self.image, 110, 255, cv2.THRESH_BINARY)

  def __convert_to_gray(self): 
    self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

  def __convert_to_hsv(self):
    self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

  def __thresold_segmentation(self):
    lower_red1 = np.array([0, 70, 50])  
    upper_red1 = np.array([12, 255, 255]) 
    lower_red2 = np.array([160, 70, 50])  
    upper_red2 = np.array([180, 255, 255]) 
    mask1 = cv2.inRange(self.image, lower_red1, upper_red1)
    mask2 = cv2.inRange(self.image, lower_red2, upper_red2)

    mask = mask1 + mask2

    # Aplicar la máscara a la imagen original
    red_segmented = cv2.bitwise_and(self.image, self.image, mask=mask)
    self.image = cv2.cvtColor(red_segmented, cv2.COLOR_BGR2RGB)


  def __closing(self, iterations):
    kernel_size = 5 
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    self.image = cv2.morphologyEx(self.image, cv2.MORPH_CLOSE, kernel, iterations=iterations)


  def __print_image(self, window_name='Image', width=1000, height=200):
    resized_image = cv2.resize(self.image, (width, height))
    cv2.imshow(window_name, resized_image)
    cv2.resizeWindow(window_name, width, height)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

  def get_segmentated_image(self):
    return self.image