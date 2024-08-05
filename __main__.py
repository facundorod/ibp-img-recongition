from preprocessing.preprocessing import PreprocessingService
from segmentation.segmentation import SegmentationService
import sys
import os

def main():
    preprocessing_service = PreprocessingService()
    preprocessing_service.start_process('images/monitor1.jpg')
    image_processed = preprocessing_service.get_image()
    segmentation_service = SegmentationService(image_processed)
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