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

    def __init__(self, model_path=None, device='CPU', classes_path=None):
        
        # Add code for Inference Engine initialization
        self.core = Core()
        
        # Add code for model loading
        self.model = self.core.read_model(model=model_path)
        self.exec_model = self.core.compile_model(model=self.model, device_name=device)

        # Add code for classes names loading
        if classes_path:
            self.classes = [line.rstrip('\n') for line in open(classes_path)]

        return

    def get_top(self, prob, topN=1):
        result = []
        
        # Add code for getting top predictions
        temp = prob
        temp = np.argsort(temp[0])
        temp = temp[-1:-topN - 1:-1]


        for k in range(0, 5):
            result.append([self.classes[temp[k]], prob[0, temp[k]]])


        return result

    def _prepare_image(self, image, h, w):
    
        # Add code for image preprocessing
        image = cv2.resize(image, (w, h))
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, axis = 0)
        
        return image

    def classify(self, image):
        probabilities = None
        
        # Add code for image classification using Inference Engine
        input_layer = self.exec_model.input(0)
        output_layer = self.exec_model.output(0)
        n, c, h, w = input_layer.shape
        image = self._prepare_image(image, h, w)

        probabilities = self.exec_model([image])[output_layer]
       

        
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
    ie_classifier = InferenceEngineClassifier(model_path=args.model, classes_path=args.classes)

    
    # Read image
    img = cv2.imread(args.input)
        
    # Classify image
    prob = ie_classifier.classify(img)
    
    # Get top 5 predictions
    predictions = ie_classifier.get_top(prob, topN=5)
    
    # print result
    log.info("Predictions: " + str(predictions))

    return


if __name__ == '__main__':
    sys.exit(main())
