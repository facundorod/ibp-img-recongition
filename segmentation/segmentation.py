import cv2
import numpy as np
import matplotlib.pyplot as plt


class SegmentationService():
  def __init__(self, image, axis_points, axis_points_values): 
    self.image = image
    self.axis_points = axis_points
    self.original = image
    self.interpolated_points = []
    self.axis_points_values = axis_points_values

  def start_process(self):
    self.__convert_to_hsv()
    self.__thresold_segmentation()
    self.__gaussian_filter()
    self.__convert_to_gray()
    self.__convert_to_binary()
    self.__skeletonize_image()
    self.__print_image(image=self.image, window_name="Imagen esqueletizada")
    self.__map_to_real_values()
    self.__graph_points()

  def __skeletonize_image(self) -> None:
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False
    size = np.size(self.image)
    skel = np.zeros(self.image.shape, np.uint8)
    while (not done):
      eroded = cv2.erode(self.image, element)
      temp = cv2.dilate(eroded, element)
      temp = cv2.subtract(self.image, temp)
      skel = cv2.bitwise_or(skel, temp)
      self.image = eroded.copy()
      zeros = size - cv2.countNonZero(self.image)

      if (zeros == size):
        done = True
    self.image = skel

  def __map_to_real_values(self) -> None:
    x_min = self.axis_points[0][0]
    x_max = self.axis_points[1][0]
    y_min = self.axis_points[2][1]
    y_max = self.axis_points[3][1]

    real_xmin = self.axis_points_values['minX']
    real_xmax = self.axis_points_values['maxX']
    real_ymin = self.axis_points_values['minY']
    real_ymax = self.axis_points_values['maxY']

    skel_points = np.column_stack(np.nonzero(self.image)) 
    x_values_skel = skel_points[:, 1]

    _, unique_indices = np.unique(x_values_skel, return_index=True)

    dataset = skel_points[unique_indices]

    real_x_points = np.interp(dataset[:, 1], [x_min, x_max],  [real_xmin, real_xmax] )

    real_y_points = np.interp(dataset[:, 0], [y_max, y_min],  [real_ymax, real_ymin])
    
    self.interpolated_points = np.column_stack((real_x_points, real_y_points))


  def __gaussian_filter(self):
    self.image = cv2.GaussianBlur(self.image, (5,5), 0)
    
  def __convert_to_binary(self):
    _, self.image = cv2.threshold(self.image, 100, 255, 0)

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


  def __print_image(self, image, window_name='Image', width=1000, height=200):
    resized_image = cv2.resize(image, (width, height))
    cv2.imshow(window_name, resized_image)
    cv2.resizeWindow(window_name, width, height)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

  def get_segmentated_image(self):
    return self.image
  
  def get_interpolated_points(self):
    return self.interpolated_points
  

  def __graph_points(self) -> None:
    plt.figure(figsize=(8, 6))
    x = self.interpolated_points[:, 0]
    y = self.interpolated_points[:, 1]
    plt.plot(x, y)
    plt.title('Curva generada a través de puntos interpolados')
    plt.show()