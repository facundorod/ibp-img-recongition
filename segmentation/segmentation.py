import cv2
import numpy as np
class SegmentationService():
  def __init__(self, image, axis_points): 
    self.image = image
    self.axis_points = axis_points
    self.contour_points = []
    self.original = image

  def start_process(self):
    self.__convert_to_hsv()
    self.__thresold_segmentation()
    self.__convert_to_gray()
    self.__convert_to_binary()
    self.__watershed()
    self.__print_image()
    
  def __convert_to_binary(self):
    _, self.image = cv2.threshold(self.image, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

  def __convert_to_gray(self): 
    self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

  def __convert_to_hsv(self):
    self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

  def __thresold_segmentation(self):
    lower_red1 = np.array([0, 60, 100])  
    upper_red1 = np.array([10, 255, 255]) 
    lower_red2 = np.array([160, 60, 100])  
    upper_red2 = np.array([180, 255, 255]) 
    mask1 = cv2.inRange(self.image, lower_red1, upper_red1)
    mask2 = cv2.inRange(self.image, lower_red2, upper_red2)

    mask = mask1 + mask2

    # Aplicar la máscara a la imagen original
    red_segmented = cv2.bitwise_and(self.image, self.image, mask=mask)
    self.image = cv2.cvtColor(red_segmented, cv2.COLOR_BGR2RGB)

  def __watershed(self):
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(self.image, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=2)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    cv2.imshow("Dist transform", dist_transform)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    # Marker labelling
    markers = cv2.connectedComponents(sure_fg)[1]
    markers = markers + 1
    markers[unknown == 255] = 0
    self.image = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
    markers = cv2.watershed(self.original, markers)
    self.original[markers == -1] = [0, 255, 255]
  

  def __print_image(self, window_name='Image', width=1000, height=200):
    resized_image = cv2.resize(self.image, (width, height))
    cv2.imshow(window_name, resized_image)
    cv2.resizeWindow(window_name, width, height)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

  def get_segmentated_image(self):
    return self.image