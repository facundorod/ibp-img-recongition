from preprocessing.preprocessing import PreprocessingService
from segmentation.segmentation import SegmentationService
from feature_measurement.feature_measurement import FeatureMeasurement
import sys
import os

def main():
    preprocessing_service = PreprocessingService()
    preprocessing_service.start_process('images/monitor2.png')
    image_processed = preprocessing_service.get_image()
    axis_points = preprocessing_service.get_axis_selected_points()
    axis_points_values = preprocessing_service.get_axis_selected_values()
    segmentation_service = SegmentationService(image_processed, axis_points, axis_points_values)
    segmentation_service.start_process()
    curve_points = segmentation_service.get_interpolated_points()
    feature_measurement = FeatureMeasurement(curve_points=curve_points)
    feature_measurement.start_process()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
            raise