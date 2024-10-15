from typing import Any
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C
import numpy as np
import scipy.signal
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

Y_LABEL = 'Presión arterial (mmHg)'
X_LABEL = 'Tiempo (s)'
class FeatureMeasurement:
  def __init__(self, curve_points) -> None:
    self.curve_points = curve_points
    self.cardiac_cycles = []
    self.peaks = None
    self.valleys = None
    self.pulse_pressure = []
    
  def start_process(self) -> None:
    self.__smooth_curve_dataset_savgol()
    # self.__smooth_curve_dataset_fourier()
    self.__detect_peaks_and_valleys()
    self.__segment_curve_by_peaks()
    self.__calculate_pulse_pressure()
    self.__plot_combined_info()
  
  def __calculate_areas(self) -> None:

    for i, segment in enumerate(self.cardiac_cycles):
        area = self.__calculate_auc(segment=segment)        
        print(f"Área bajo el pulso {i + 1}: {area:.2f}")
  

  def __smooth_curve_dataset_savgol(self) -> None:
    X = self.curve_points[:, 0]
    y = self.curve_points[:, 1]

    smoothed_data_savgol = savgol_filter(y, window_length=30, polyorder=8)
    self.x_points = X
    self.y_points = smoothed_data_savgol
    self.curve_points = np.column_stack((X, smoothed_data_savgol))


  def __detect_peaks_and_valleys(self) -> None:
    mask = ~np.isnan(self.curve_points[:, 1])
    self.x_values_masked = self.curve_points[:, 0][mask]
    self.y_values_masked = self.curve_points[:, 1][mask]
    self.masked_point = np.column_stack((self.x_values_masked, self.y_values_masked))

    # Detect peaks and valleys on the masked y values
    peaks_index = scipy.signal.argrelextrema(self.y_values_masked, np.greater, order=80)[0]
    valleys_index = scipy.signal.argrelextrema(self.y_values_masked, np.less, order=80)[0]

    self.peaks = peaks_index
    self.valleys = valleys_index

  def __segment_curve_by_peaks(self) -> None:
    cardiac_cycles = []

    for i in range(len(self.valleys) - 1):
        cycle_start = self.valleys[i]
        cycle_end = self.valleys[i + 1]
        
        cycle_points = self.masked_point[cycle_start:cycle_end]
        cardiac_cycles.append(cycle_points)

    self.cardiac_cycles = cardiac_cycles

  def __calculate_pulse_pressure(self):
    for cardiac_cycle in self.cardiac_cycles:
      y_values = np.array(cardiac_cycle[:, 1])
      systolic_value = y_values.max()
      diastolic_value = y_values.min()
      pulse_pressure = systolic_value - diastolic_value
      self.pulse_pressure.append(pulse_pressure)

  
  def __calculate_delta_pp(self): 
    pressure_values = np.array(self.pulse_pressure)
    min_index = np.argmin(pressure_values)
    max_index = np.argmax(pressure_values)

    # Retrieve the minimum and maximum pulse pressure cycles
    self.delta_pp = pressure_values[max_index] - pressure_values[min_index]
    print(f"Delta PP: {self.delta_pp}")


  # Calcular el área bajo la curva usando el método de trapezoides 
  def __calculate_auc(self, segment):
    x_values = segment[:, 0]
    y_values = segment[:, 1]
    area = np.trapz(y_values, x_values)
    return area

  def __calculate_systolic_slope(self) -> None:
    systolic_slopes = []

    for i, cardiac_cycle in enumerate(self.cardiac_cycles):
        start_x, start_y = cardiac_cycle[0, 0], cardiac_cycle[0, 1]
        
        peak_index = np.argmax(cardiac_cycle[:, 1])
        peak_x, peak_y = cardiac_cycle[peak_index, 0], cardiac_cycle[peak_index, 1]
        
        if (peak_x - start_x != 0):
          slope = (peak_y - start_y) / (peak_x - start_x)
        else:
          slope = 0
        systolic_slopes.append(slope)

        print(f"Pendiente sistólica para ciclo {i + 1}: {slope:.2f}")
    
    self.systolic_slopes = systolic_slopes

  def __plot_combined_info(self) -> None:
    plt.figure(figsize=(20, 10))
    
    plt.plot(self.curve_points[:, 0], self.curve_points[:, 1], label='Curva de presión arterial', color='gray', linestyle="--")

    pressure_values = np.array(self.pulse_pressure)
    min_index = np.argmin(pressure_values)
    max_index = np.argmax(pressure_values)

    for i, cycle in enumerate(self.cardiac_cycles):
        x_points = cycle[:, 0]
        y_points = cycle[:, 1]

        if i == min_index:
            plt.plot(x_points, y_points, color='blue', linewidth=0.5, label=f'Ciclo con menor presión ({self.pulse_pressure[min_index]:.2f})', marker='o', markersize=4)
        elif i == max_index:
            plt.plot(x_points, y_points, color='red', linewidth=0.5, label=f'Ciclo con mayor presión ({self.pulse_pressure[max_index]:.2f})', marker='s', markersize=4)
        y_min_limit = y_points.min()
        plt.fill_between(x_points, y_points, y_min_limit, alpha=0.3)  
        start_x, start_y = x_points[0], y_points[0]
        peak_index = np.argmax(y_points)
        peak_x, peak_y = x_points[peak_index], y_points[peak_index]
        plt.plot([start_x, peak_x], [start_y, peak_y], linestyle="--", color="green", markersize=8)

    custom_lines = [
        Line2D([0], [0], color='blue', marker='o', markersize=6, label=f'Ciclo con menor presión ({self.pulse_pressure[min_index]:.2f})'),
        Line2D([0], [0], color='red', marker='s', markersize=6, label=f'Ciclo con mayor presión ({self.pulse_pressure[max_index]:.2f})'),
        Line2D([0], [0], color="green", linestyle="--", label="dp/dt")
    ]
    plt.legend(handles=custom_lines, loc="best")

    # Calcular y anotar Delta PP
    delta_pp = self.pulse_pressure[max_index] - self.pulse_pressure[min_index]
    plt.text(0.05, 0.95, f'Delta PP: {delta_pp:.2f}', horizontalalignment='right', verticalalignment='top',
             transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.6))

    # Etiquetas de los ejes y título
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Presión arterial (mmHg)")
    plt.title('Análisis de Ciclos Cardíacos: Ciclos con Menor y Mayor Presión, Delta PP, y Pendiente de Sístole')

    plt.show()