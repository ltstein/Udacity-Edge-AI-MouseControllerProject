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
        # print("Initializing Gaze Estimation Model...")
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

    def predict(self, left_eye, right_eye, head_angles):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        head_pose_angle, left_eye_input, right_eye_input = self.preprocess_input(left_eye, right_eye, head_angles)
        input_dict = {'head_pose_angles':head_pose_angle,'left_eye_image':left_eye_input,'right_eye_image':right_eye_input}
        
        start_time = time.time()
        self.net.start_async( request_id = 0, inputs=input_dict)
        status = self.net.requests[0].wait(-1)

        if status == 0:
            output = self.net.requests[0].outputs[self.output_name]
            self.infer_time = time.time() - start_time
        return output

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

        #Create input blobs
        p_l_eye = cv2.resize(l_eye, (60,60), interpolation=cv2.INTER_AREA)
        p_l_eye = p_l_eye.transpose((2, 0, 1))
        l_input_blob = p_l_eye.reshape(1, 3, 60, 60)

        p_r_eye = cv2.resize(r_eye, (60,60), interpolation=cv2.INTER_AREA)
        p_r_eye = p_r_eye.transpose((2, 0, 1))
        r_input_blob = p_r_eye.reshape(1, 3, 60,60)

        head_input_blob = np.array([[head_angles[0], head_angles[1], head_angles[2]]])

        return head_input_blob, l_input_blob, r_input_blob

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
