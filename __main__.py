from preprocessing.preprocessing import PreprocessingService
from segmentation.segmentation import SegmentationService
import sys
import os

def main():
    preprocessing_service = PreprocessingService()
    preprocessing_service.start_process('images/monitor.png')
    image_processed = preprocessing_service.get_image()
    axis_points = preprocessing_service.get_axis_selected_points()
    axis_points_values = preprocessing_service.get_axis_selected_values()
    segmentation_service = SegmentationService(image_processed, axis_points, axis_points_values)
    segmentation_service.start_process()

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