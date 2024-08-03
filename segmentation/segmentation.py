import cv2
import numpy as np
class SegmentationService():
  def __init__(self, image): 
    self.image = image

  def start_process(self):
    # self.__convert_to_gray()
    self.__convert_to_hsv()
    self.__thresold_segmentation()
    self.__closing()
    self.__print_image(window_name="Filtro de Canny para detección de bordes")

  def __convert_to_gray(self): 
    self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

  def __convert_to_hsv(self):
    self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
 

  def __thresold_segmentation(self):
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(self.image, lower_red1, upper_red1)
    mask2 = cv2.inRange(self.image, lower_red2, upper_red2)

    mask = mask1 + mask2

    # Aplicar la máscara a la imagen original
    red_segmented = cv2.bitwise_and(self.image, self.image, mask=mask)
    self.image = cv2.cvtColor(red_segmented, cv2.COLOR_BGR2RGB)

  def __opening(self): 
    # Definir el kernel para la erosión
    kernel_size = 5 
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    self.image = cv2.morphologyEx(self.image, cv2.MORPH_OPEN, kernel)


  def __closing(self):
    # Definir el kernel para la erosión
    kernel_size = 5 
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    self.image = cv2.morphologyEx(self.image, cv2.MORPH_CLOSE, kernel)

  def __sobel_filter(self):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    self.image = cv2.magnitude(sobel_x, sobel_y)

  def __prewitt_filter(self):
    kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    img_prewittx = cv2.filter2D(image, -1, kernelx)
    img_prewitty = cv2.filter2D(image, -1, kernely)
    self.image = img_prewittx + img_prewitty

  def __canny_filter(self):
    umbral_thresold_initial = 80
    kernel_size = 3
    umbral_thresold_end = 200
    self.image = cv2.Canny(self.image, umbral_thresold_initial, umbral_thresold_end, apertureSize = kernel_size,
      L2gradient=True)

  def __print_image(self, window_name='Image', width=1500, height=600):
    resized_image = cv2.resize(self.image, (width, height))
    cv2.imshow(window_name, resized_image)
    cv2.resizeWindow(window_name, width, height)
    cv2.waitKey(0)
    cv2.destroyAllWindows()