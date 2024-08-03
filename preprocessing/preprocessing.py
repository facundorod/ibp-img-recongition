import cv2
import numpy as np

class PreprocessingService:
  def __init__(self):
    self.image = None
    self.display_image = None 
    self.warped_image = None
    self.points = []
    self.window_width = 1500
    self.window_height = 900

  def start_process(self, image_path):
    self.__open_image(image_path)
    if self.image is not None:
        self.__clip_image()
        self.__crop_image()

  def __open_image(self, image_path):
    self.image = cv2.imread(image_path)
    if self.image is None:
        print("Error: Could not open or find the image.")
    else:
        print("Image loaded successfully.")

  def __clip_image(self):
    self.display_image = self.image.copy()
    resized_image = cv2.resize(self.display_image, (self.window_width, self.window_height))
    cv2.imshow("Image", resized_image)
    cv2.setMouseCallback("Image", self.__select_points)

    # Esperar hasta que se seleccionen 4 puntos
    print("Please click on 4 points on the image.")
    while len(self.points) < 4:
        cv2.waitKey(1)

    cv2.destroyAllWindows()

    # Imprimir los puntos seleccionados
    print("Selected points:")
    for i, point in enumerate(self.points):
        print(f"Point {i + 1}: {point}")

  def __select_points(self, event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
      # Ajuste de escala para la imagen redimensionada
      scale_x = self.image.shape[1] / self.window_width
      scale_y = self.image.shape[0] / self.window_height
      x_new = int(x * scale_x)
      y_new = int(y * scale_y)
      self.points.append((x_new, y_new))
      cv2.circle(self.display_image, (x_new, y_new), 10, (0, 255, 0), -1)
      # Redibuja la imagen después de agregar cada punto
      resized_image = cv2.resize(self.display_image, (self.window_width, self.window_height))
      cv2.imshow("Image", resized_image)
      print(f"Point selected: ({x_new}, {y_new})")

  def __crop_image(self):
    if len(self.points) == 4:
      # Ordenar los puntos en el orden: top-left, top-right, bottom-right, bottom-left
      self.points = self.__order_points(self.points)
      # El tamaño del rectángulo objetivo
      width, height = self.__calculate_new_dimensions(self.points)
      self.__perspective_correction(width, height)
    else:
      print("Error: No se seleccionaron 4 puntos correctamente.")


  def __perspective_correction(self, width, height):
    # Puntos de destino (la nueva imagen "rectangular")
    pts_target = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype="float32")
    # Obtener la matriz de transformación de perspectiva
    pts_original = np.array(self.points, dtype="float32")
    matrix = cv2.getPerspectiveTransform(pts_original, pts_target)

    # Aplicar la transformación de perspectiva
    self.warped_image = cv2.warpPerspective(self.image, matrix, (width, height))
    # Mostrar la imagen recortada
    cv2.imshow("Cropped Image", self.warped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

  def __order_points(self, points):
    # Ordenar los puntos: top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    s = np.sum(points, axis=1)
    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]

    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]

    return rect

  def __calculate_new_dimensions(self, points):
    # Calcula el ancho y la altura del nuevo rectángulo
    (tl, tr, br, bl) = points

    # Calcular el ancho del nuevo rectángulo
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    # Calcular la altura del nuevo rectángulo
    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    return max_width, max_height


  def get_image(self): 
    return self.warped_image


