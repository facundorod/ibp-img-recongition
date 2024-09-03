from typing import Any
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
class FeatureMeasurement:
  def __init__(self, curve_points) -> None:
    self.curve_points = curve_points
    self.pulse_segments = []
    self.peaks = None


  def start_process(self) -> None:
    self.__smooth_curve()
    self.__detect_peaks()
    self.__segment_curve_by_peaks()
    self.__calculate_areas()
    self.__calculate_delta_pp()

  
  def __calculate_areas(self) -> None:
    areas = [self.__calculate_auc(segment) for segment in self.pulse_segments]
    for i, area in enumerate(areas):
      print(f"Área bajo el pulso {i + 1}: {area}")

  def __smooth_curve(self) -> None:
    X = self.curve_points[:, 0]
    y = self.curve_points[:, 1]
    kernel = C(1.0, (1e-4, 1e1)) * RBF(1, (1e-4, 1e1))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-2, random_state=0)
    gp.fit(X.reshape(-1, 1), y)
    x_pred = np.linspace(X.min(), X.max(), 1000).reshape(-1, 1)
    y_pred, _ = gp.predict(x_pred, return_std=True)
    plt.figure()
    plt.plot(x_pred, y_pred)
    plt.xlabel('Posición/tiempo')
    plt.ylabel('Presión arterial')
    plt.title("Puntos luego de proceso gauseano de regresión")
    self.x_points = x_pred
    self.y_points = y_pred
    self.curve_points = np.column_stack((self.x_points, self.y_points))
    plt.show()


  def __detect_peaks(self) -> None:
    peaks, _ = find_peaks(self.y_points, distance=50) 
    self.peaks = peaks 

  def __segment_curve_by_peaks(self) -> None:
    pulse_segments = []
    for i in range(len(self.peaks) - 1):
        start = self.peaks[i]
        end = self.peaks[i + 1]
        pulse_segments.append(self.curve_points[start:end])
    pulse_segments.append(self.curve_points[self.peaks[-1]:])
    self.pulse_segments = pulse_segments


  def __calculate_delta_pp(self):
    intervals = np.diff(self.x_points[self.peaks])
    pulse_frequencies = 1 / intervals
    delta_pp = np.diff(pulse_frequencies)
    for i in range(len(delta_pp)):
      print(f"Frecuencia de Pulso {i + 1}: {pulse_frequencies[i]} Hz")
      print(f"Delta PP entre Pulso {i + 1} y Pulso {i + 2}: {delta_pp[i]} Hz")

  def __calculate_auc(self, segment):
    x_values = segment[:, 0]
    y_values = segment[:, 1]
    area = np.trapz(y_values, x_values)
    return area
