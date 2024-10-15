import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
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
    self.__convert_to_gray()
    self.__gaussian_filter()
    self.__convert_to_binary()
    self.__skeletonize_image()
    self.__map_to_real_values()

  def __skeletonize_image(self) -> None:    
    self.image = skeletonize(self.image)
    self.image = (self.image * 255).astype(np.uint8)
    self.__print_image(image = self.image)
    

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
    x_values = dataset[:, 1]
    index_discontinuity = self.__detect_discontinuity(x_values)

    print(f"index_discontinuity {index_discontinuity}")

    skel_points_left = dataset[:index_discontinuity + 1]
    skel_points_right = dataset[index_discontinuity + 1:]

    real_x_left = np.interp(skel_points_left[:, 1], [x_min, x_max], [real_xmin, real_xmax])
    real_y_left = np.interp(skel_points_left[:, 0], [y_max, y_min], [real_ymax, real_ymin])

    real_x_right = np.interp(skel_points_right[:, 1], [x_min, x_max], [real_xmin, real_xmax])
    real_y_right = np.interp(skel_points_right[:, 0], [y_max, y_min], [real_ymax, real_ymin])

    x_smooth = np.concatenate((real_x_left, [np.nan], real_x_right))
    y_smooth = np.concatenate((real_y_left, [np.nan], real_y_right))

    self.interpolated_points = np.column_stack((x_smooth, y_smooth))


  def __gaussian_filter(self): 
    self.image = cv2.GaussianBlur(self.image, (5,5), 0)
    
  def __convert_to_binary(self):
    _, self.image = cv2.threshold(self.image, 90, 255, 0)

  def __convert_to_gray(self): 
    self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)


  def __convert_to_hsv(self):
    self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

  def __thresold_segmentation(self):
    lower_red1 = np.array([0, 80, 100])  
    upper_red1 = np.array([10, 255, 255]) 
    lower_red2 = np.array([150, 80, 100])  
    upper_red2 = np.array([180, 255, 255]) 
    mask1 = cv2.inRange(self.image, lower_red1, upper_red1)
    mask2 = cv2.inRange(self.image, lower_red2, upper_red2)

    mask = mask1 + mask2

    red_segmented = cv2.bitwise_and(self.image, self.image, mask=mask)
    self.image = cv2.cvtColor(red_segmented, cv2.COLOR_BGR2RGB)


  def __detect_discontinuity(self, values):
    x_differences = np.abs(np.diff(values))

    index_max_difference = np.argmax(x_differences)

    if (index_max_difference):
      return index_max_difference
    
    return None

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