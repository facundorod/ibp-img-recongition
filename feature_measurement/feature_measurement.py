from typing import Any
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C
import numpy as np
import scipy.signal
from scipy.fft import rfft, irfft, rfftfreq
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
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
    self.__calculate_areas()
    self.__calculate_pulse_pressure()
    self.__calculate_delta_pp()
  
  def __calculate_areas(self) -> None:
    plt.plot(self.curve_points[:, 0], self.curve_points[:, 1], label='Curva de presión arterial')

    for i, segment in enumerate(self.cardiac_cycles):
        x_points = segment[:, 0]
        y_points = segment[:, 1]
        
        area = self.__calculate_auc(segment=segment)

        y_min_limit = y_points.min()

        plt.fill_between(x_points, y_points, y_min_limit, alpha=0.3, label=f'Área {area:.2f}')
        
        print(f"Área bajo el pulso {i + 1}: {area:.2f}")

    plt.xlabel(X_LABEL)
    plt.ylabel(Y_LABEL)
    plt.title('Áreas bajo la curva de cada ciclo cardíaco')
    plt.legend()
    plt.show()

  

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

    # Plot the curve with peaks and valleys marked
    plt.plot(self.curve_points[:, 0], self.curve_points[:, 1], label='Curva de presión arterial')
    plt.plot(self.x_values_masked[peaks_index], self.y_values_masked[peaks_index], 'ro', label='Picos detectados')
    plt.plot(self.x_values_masked[valleys_index], self.y_values_masked[valleys_index], 'bo', label='Valles detectados')
    plt.legend()
    plt.xlabel(X_LABEL)
    plt.ylabel(Y_LABEL)
    plt.title('Mínimos y máximos detectados en la curva de presión arterial')
    plt.show()

    self.peaks = peaks_index
    self.valleys = valleys_index

  def __segment_curve_by_peaks(self) -> None:
    cardiac_cycles = []

    plt.plot(self.x_values_masked, self.y_values_masked, label='Curva de presión arterial')

    # Loop through each cycle based on valleys and plot them separately
    for i in range(len(self.valleys) - 1):
        cycle_start = self.valleys[i]
        cycle_end = self.valleys[i + 1]
        
        cycle_points = self.masked_point[cycle_start:cycle_end]
        cardiac_cycles.append(cycle_points)

        plt.plot(cycle_points[:, 0], cycle_points[:, 1], label=f'Ciclo {i + 1}')

    plt.xlabel(X_LABEL)
    plt.ylabel(Y_LABEL)
    plt.title('Ciclos cardíacos')
    plt.legend()
    plt.show()


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
    min_cycle = self.cardiac_cycles[min_index]
    max_cycle = self.cardiac_cycles[max_index]

    # Separate x and y values for each cycle
    min_x, min_y = min_cycle[:, 0], min_cycle[:, 1]
    max_x, max_y = max_cycle[:, 0], max_cycle[:, 1]

    plt.figure(figsize=(10, 6))
    
    # Plot the cycle with the minimum pulse pressure
    plt.plot(min_x, min_y, label=f'Ciclo de menor presión de pulso ({self.pulse_pressure[min_index]:.2f})', color='blue')

    # Plot the cycle with the maximum pulse pressure
    plt.plot(max_x, max_y, label=f'Ciclo de mayor presión de pulso ({self.pulse_pressure[max_index]:.2f})', color='red')
    
    # Calculate delta PP and display it on the plot
    self.delta_pp = pressure_values[max_index] - pressure_values[min_index]
    print(f"Delta PP: {self.delta_pp}")

    # Annotate delta PP on the plot
    plt.text(0.05, 0.95, f'Delta PP: {self.delta_pp:.2f}', horizontalalignment='right',
             verticalalignment='center', transform=plt.gca().transAxes,
             fontsize=12, bbox=dict(facecolor='white', alpha=0.6))

    # Axis labels and plot title
    plt.xlabel(X_LABEL)  
    plt.ylabel(Y_LABEL)
    plt.title('Ciclos Cardíacos con Mayor y Menor Amplitud de Presión de Pulso')
    plt.legend()
    plt.show()


  # Calcular el área bajo la curva usando el método de trapezoides 
  def __calculate_auc(self, segment):
    x_values = segment[:, 0]
    y_values = segment[:, 1]
    area = np.trapz(y_values, x_values)
    return area

