"""
Classification sample

Command line to run:
python doge_classifier.py -i image.jpg \
    -m mobilenet-v2-pytorch.xml -c imagenet_synset_words.txt
"""

import os
import cv2
import sys
import argparse
import numpy as np
import logging as log
from openvino.runtime import Core


class InferenceEngineClassifier:

    def __init__(self, model_path, device='CPU', classes_path=None):
        
        # Add code for Inference Engine initialization
        
        # Add code for model loading

        # Add code for classes names loading
        
        return

    def get_top(self, prob, topN=1):
        result = []
        
        # Add code for getting top predictions
        
        return result

    def _prepare_image(self, image, h, w):
    
        # Add code for image preprocessing
        
        return image

    def classify(self, image):
        probabilities = None
        
        # Add code for image classification using Inference Engine
        
        return probabilities


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
    return parser


def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s",
        level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()

    log.info("Start IE classification sample")

    # Create InferenceEngineClassifier object
    
    # Read image
        
    # Classify image
    
    # Get top 5 predictions
    
    # print result

    return


if __name__ == '__main__':
    sys.exit(main())
