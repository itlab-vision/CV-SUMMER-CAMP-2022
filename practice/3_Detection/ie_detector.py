"""
Object detector based on Inference Engine
"""
import os
import sys

sys.path.append(r'D:\_dev\open_model_zoo\demos\common\python')

import cv2
import numpy as np
from openvino.runtime import Core

import logging as log
import argparse
import pathlib
from time import perf_counter

from openvino.model_zoo.model_api.models import DetectionModel, DetectionWithLandmarks, RESIZE_TYPES, OutputTransform
from openvino.model_zoo.model_api.performance_metrics import PerformanceMetrics
from openvino.model_zoo.model_api.pipelines import get_user_config, AsyncPipeline
from openvino.model_zoo.model_api.adapters import create_core, OpenvinoAdapter
from images_capture import open_images_capture


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='Path to an .xml \
        file with a trained model.', required=True, type=str)
    parser.add_argument('-i', '--input', help='Path to \
        image file', required=True, type=str)
    parser.add_argument('-d', '--device', help='Specify the target \
        device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. \
        Sample will look for a suitable plugin for device specified \
        (CPU by default)', default='CPU', type=str)
    parser.add_argument('-c', '--classes', help='File containing classes \
        names', type=str, default=None)
    parser.add_argument('-t', '--prob_threshold', default=0.5, type=float,
        help='Optional. Probability threshold for detections filtering.')
    return parser
  
  
def draw_detections(frame, detections, labels, threshold):

    return frame

def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s",
                    level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()

    log.info("Start IE detection sample")

    # Initialize data input

    
    
    # Initialize OpenVINO

    
    # Initialize Plugin configs

    
    #Load SSD model

    
    # Initialize pipeline

    while True:

        # Get one image

        # Start processing frame asynchronously
    
    
        # Draw detections in the image
    
        # Show image and wait for key press
        
        # Wait 1 ms and check pressed button to break the loop
        pass

        
    # Destroy all windows
    cv2.destroyAllWindows()
    return


if __name__ == '__main__':
    sys.exit(main())
