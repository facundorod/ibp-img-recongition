from typing import Any
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import numpy as np
import scipy.signal
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
class FeatureMeasurement:
  def __init__(self, curve_points) -> None:
    self.curve_points = curve_points
    self.cardiac_cycles = []
    self.peaks = None
    self.valleys = None

  def start_process(self) -> None:
    self.__smooth_curve_dataset()
    self.__detect_peaks_and_valleys()
    self.__segment_curve_by_peaks()
    # self.__calculate_areas()
    # self.__calculate_delta_pp()

  
  def __calculate_areas(self) -> None:
    areas = [self.__calculate_auc(segment) for segment in self.pulse_segments]
    for i, area in enumerate(areas):
      print(f"Área bajo el pulso {i + 1}: {area}")

  def __smooth_curve_dataset(self) -> None:
    X = self.curve_points[:, 0]
    y = self.curve_points[:, 1]
    kernel = C(1.0, (1e-3, 1e1)) * RBF(1, (1e-2, 1e1))

    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-3, random_state=0)
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


  def __detect_peaks_and_valleys(self) -> None:
    peaks_index = scipy.signal.argrelextrema(self.y_points, np.greater, order=50)[0]
    valleys_index = scipy.signal.argrelmin(self.y_points, order=50)[0]
    plt.plot(self.x_points, self.y_points, label='Curva de presión arterial')
    plt.plot(self.x_points[peaks_index], self.y_points[peaks_index], 'ro', label='Picos detectados')
    plt.plot(self.x_points[valleys_index], self.y_points[valleys_index], 'bo', label='Valles detectados')
    plt.legend()
    plt.xlabel('Posición/tiempo')
    plt.ylabel('Presión arterial')
    plt.title('Picos y valles detectados en la curva de presión arterial')
    plt.show()
    self.peaks = peaks_index 
    self.valleys = valleys_index
    

  def __segment_curve_by_peaks(self) -> None:
    cardiac_cycles = []

    for i in range(len(self.valleys) - 1):
        cycle_start = self.valleys[i]
        cycle_end = self.valleys[i + 1]
        cycle_points = self.curve_points[cycle_start:cycle_end]
        cardiac_cycles.append(cycle_points)

        plt.figure()
        plt.plot(cycle_points[:, 0], cycle_points[:, 1], label=f'Ciclo cardíaco {i + 1}')
        plt.xlabel('Posición/tiempo')
        plt.ylabel('Presión arterial')
        plt.title(f'Ciclo Cardíaco {i + 1} (Valle a Valle)')
        plt.legend()
        plt.show()

    self.cardiac_cycles = cardiac_cycles

  def __calculate_delta_pp(self):
    # Encuentra la diferencia de tiempo entre picos consecutivos.
    dy = np.diff(self.y_points) / np.diff(self.x_points)
    change_indices = np.where(np.diff(np.sign(dy)))[0] + 1 
    intervals = np.diff(self.x_points[change_indices])
    pulse_frequencies = 1 / intervals
    delta_pp = np.diff(pulse_frequencies)

    for i in range(len(delta_pp)):
      print(f"Frecuencia de Pulso {i + 1}: {pulse_frequencies[i]} Hz")
      print(f"Delta PP entre Pulso {i + 1} y Pulso {i + 2}: {delta_pp[i]} Hz")

    return delta_pp

  # Calcular el área bajo la curva usando el método de trapezoides 
  def __calculate_auc(self, segment):
    x_values = segment[:, 0]
    y_values = segment[:, 1]
    area = np.trapz(y_values, x_values)
    return area
