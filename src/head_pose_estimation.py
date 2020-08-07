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


class HeadPoseEstimator:
    '''
    Class for the Head Pose Estimation Model.
    '''

    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.model_weights = model_name+'.bin'
        self.model_structure = model_name+'.xml'
        self.device = device
        # Check if network can be initialized.
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
        start_time = time.time()
        core = IECore()
        self.net = core.load_network(
            network=self.model, device_name=self.device, num_requests=1)
        self.load_time = time.time() - start_time

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        p_frame = self.preprocess_input(image)
        input_dict = {self.input_name: p_frame}
        start_time = time.time()

        self.net.start_async(request_id=0, inputs=input_dict)
        status = self.net.requests[0].wait(-1)

        if status == 0:
            output = self.net.requests[0].outputs
            self.infer_time = time.time() - start_time
            yaw, pitch, roll = self.preprocess_output(output)
        return yaw, pitch, roll

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.

        name: "data" , shape: [1x3x60x60] - An input image in [1xCxHxW] format. Expected color order is BGR.

        '''
        start_time = time.time()
        p_image = cv2.resize(
            image, (self.input_shape[3], self.input_shape[2]), interpolation=cv2.INTER_AREA)
        p_image = p_image.reshape(
            1, 3, self.input_shape[2], self.input_shape[3])
        self.preprocess_input_time = time.time() - start_time
        return p_image

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.

        Output layer names in Inference Engine format:

        name: "angle_y_fc", shape: [1, 1] - Estimated yaw (in degrees).
        name: "angle_p_fc", shape: [1, 1] - Estimated pitch (in degrees).
        name: "angle_r_fc", shape: [1, 1] - Estimated roll (in degrees).

        Output layer names in Caffe* format:

        name: "fc_y", shape: [1, 1] - Estimated yaw (in degrees).
        name: "fc_p", shape: [1, 1] - Estimated pitch (in degrees).
        name: "fc_r", shape: [1, 1] - Estimated roll (in degrees).

        Each output contains one float value that represents value in Tait-Bryan angles (yaw, pitch or roll).

        '''
        return outputs['angle_y_fc'], outputs['angle_p_fc'], outputs['angle_r_fc']
