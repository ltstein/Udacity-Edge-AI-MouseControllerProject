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

class GazeEstimator:
    '''
    Class for the Gaze Estimation Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        print("Initializing Gaze Estimation Model...")
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
        core = IECore()
        self.net = core.load_network(
            network=self.model, device_name=self.device, num_requests=1)

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        raise NotImplementedError

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, l_eye, r_eye, head_angles):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        
        
        Blob in the format [BxCxHxW] where:
            B - batch size
            C - number of channels
            H - image height
            W - image width

        with the name left_eye_image and the shape [1x3x60x60].

        Blob in the format [BxCxHxW] where:
            B - batch size
            C - number of channels
            H - image height
            W - image width

        with the name right_eye_image and the shape [1x3x60x60].

        Blob in the format [BxC] where:
            B - batch size
            C - number of channels

        with the name head_pose_angles and the shape [1x3].
        '''
        left = l_eye.reshape

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        
        The net outputs a blob with the shape: [1, 3], containing Cartesian coordinates of gaze direction vector. Please note that the output vector is not normalizes and has non-unit length.

        Output layer name in Inference Engine format:

        gaze_vector

        Output layer name in Caffe2 format:

        gaze_vector
        
        '''
        raise NotImplementedError
