'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

import sys
import argparse
import cv2
import os
from openvino.inference_engine import IENetwork, IECore
import time
import numpy as np

class FaceDetector:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        # print("Initializing Face Detection Model...")
        self.model_weights = model_name+'.bin'
        self.model_structure = model_name+'.xml'
        self.device = device
        # self.threshold = threshold
        # Check if network can be initialized. TODO: Is this deprecated?
        try:
            self.model = IENetwork(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError(
                "Could not Initialise the network. Have you enterred the correct model path?")

        self.input_name = next(iter(self.model.inputs))
        self.input_shape = self.model.inputs[self.input_name].shape
        self.output_name = next(iter(self.model.outputs))
        self.output_shape = self.model.outputs[self.output_name].shape

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        # print("Load Model")
        core = IECore()
        self.net = core.load_network(
            network=self.model, device_name=self.device, num_requests=1)

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        p_frame = self.preprocess_input(image)
        input_dict = {self.input_name: p_frame}
        start_time = time.time()
        self.net.start_async( request_id = 0, inputs=input_dict)
        status = self.net.requests[0].wait(-1)

        if status == 0:
            output = self.net.requests[0].outputs[self.output_name]
            self.infer_time = time.time() - start_time
            coords = self.preprocess_output(output, image)
            # image = self.draw_outputs(image, coords)
        return coords, image
        

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        face-detection-adas-binary-0001
        Inputs
        Name: input, shape: [1x3x384x672] - An input image in the format [BxCxHxW], where:

        B - batch size
        C - number of channels
        H - image height
        W - image width

        Expected color order is BGR.

        '''
        # print("Preprocessing input...")
        p_image = cv2.resize(image, (self.input_shape[3],self.input_shape[2]), interpolation= cv2.INTER_AREA)
        p_image = np.moveaxis(p_image, -1, 0)

        return p_image

    def preprocess_output(self, outputs, image):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.

        The net outputs blob with shape: [1, 1, N, 7], where N is the number of detected bounding boxes. Each detection has the format [image_id, label, conf, x_min, y_min, x_max, y_max], where:

        image_id - ID of the image in the batch
        label - predicted class ID
        conf - confidence for the predicted class
        (x_min, y_min) - coordinates of the top left bounding box corner
        (x_max, y_max) - coordinates of the bottom right bounding box corner.

        '''
        # print("preprocess output")
        height, width = image.shape[0:2]
        coordinates = []
        results = outputs
        for item in results[0][0]:
            conf = item[2]
            if conf >= 0.6:
                xmin = int(item[3] * width)
                ymin = int(item[4] * height)
                xmax = int(item[5] * width)
                ymax = int(item[6] * height)
                coordinates.append((xmin, ymin, xmax, ymax))
        return coordinates

    def draw_outputs(self, image, coords):
#     '''
#     TODO: This method needs to be completed by you
#     '''
        # print("Draw output")
        for box in coords:
                cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 1)
        return image
