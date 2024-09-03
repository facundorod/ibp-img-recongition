import cv2
import numpy as np
from tkinter import simpledialog
import tkinter as tk

class PreprocessingService:
  def __init__(self):
    self.image = None
    self.display_image = None 
    self.warped_image = None
    self.points = []
    self.axis_points = []
    self.window_width = 1000
    self.axis_values = {'minX': None, 'maxX': None, 'minY': None, 'maxY': None}
    self.window_height = 800

  def start_process(self, image_path):
    self.__open_image(image_path)
    if self.image is not None:
        self.__perspective_correction()
        self.__select_axis_points()
        self.__gaussean_filter()

  def __open_image(self, image_path):
    self.image = cv2.imread(image_path)
    if self.image is None:
        print("Error: Could not open or find the image.")
    else:
        print("Image loaded successfully.")

  def __select_axis_points(self):
    self.axis_image = self.warped_image.copy()
    windows_name = "Seleccione el eje x e y de la imagen"
    cv2.imshow(windows_name, self.axis_image)
    cv2.setMouseCallback(windows_name, self.__click_axis_points)

    print("Please click on 4 points on the image for axis (xMin, xMax, yMin, yMax).")
    
    while len(self.axis_points) < 4:
        cv2.waitKey(1)

    cv2.destroyAllWindows()

    print("Selected axis points:")
    for i, point in enumerate(self.axis_points):
        print(f"Axis Point {i + 1}: {point}")

    print(f"Axis values: xMin = {self.axis_values['minX']}, xMax = {self.axis_values['maxX']}, yMin = {self.axis_values['minY']}, yMax = {self.axis_values['maxY']}")

  def __click_axis_points(self, event, x, y, flags, params):
    labels = ['minX', 'maxX', 'minY', 'maxY']
    if event == cv2.EVENT_LBUTTONDOWN and len(self.axis_points) < 4:
      x_new = int(x)
      y_new = int(y)
      # Asignar el punto al eje correspondiente
      current_label = labels[len(self.axis_points)]
      self.axis_points.append((x_new, y_new))
      cv2.circle(self.axis_image, (x, y), 10, (255, 0, 0), -1)
      # Redibuja la imagen después de agregar cada punto
      cv2.imshow("Seleccione el eje x e y de la imagen", self.axis_image)
      print(f"Point selected for {current_label}: ({x_new}, {y_new})")
      # Solicitar al usuario que ingrese el valor real para este punto
      root = tk.Tk()
      root.withdraw()  # Ocultar la ventana principal de Tkinter
      value = simpledialog.askfloat("Input", f"Enter the value for {current_label}:")
      self.axis_values[current_label] = value
      print(f"Value entered for {current_label}: {value}")

      # Mostrar el siguiente mensaje de instrucción
      if len(self.axis_points) < 4:
          next_instruction = labels[len(self.axis_points)]
          print(f"Please click on the point for {next_instruction} and enter its value.")


  def __perspective_correction(self) -> None:

    width, height = self.image.shape[1], self.image.shape[0]

    pts_original = np.float32([
        [0, 0],             
        [width - 1, 0],     
        [0, height - 1],    
        [width - 1, height - 1] 
    ])

    pts_corrected = np.float32([
        [0, 0],             
        [width - 1, 0],     
        [0, height - 1],    
        [width - 1, height - 1]  
    ])
     
    matrix = cv2.getPerspectiveTransform(pts_original, pts_corrected)

    self.warped_image = cv2.warpPerspective(self.image, matrix, (width, height))


  def get_image(self):
    return self.warped_image
  
  def __gaussean_filter(self): 
    kernel = np.ones((5,5),np.float32) / 25
    self.warped_image = cv2.filter2D(self.warped_image, -1, kernel)
  
  def get_axis_selected_points(self):
    return self.axis_points

  
  def get_axis_selected_values(self):
     return self.axis_values