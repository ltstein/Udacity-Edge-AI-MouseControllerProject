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


class FacialLandmarkDetector:
    '''
    Class for the Facial Landmarks Model.
    '''

    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        print("Initializing Facial Landmarks Model...")
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
        p_frame, p_image = self.preprocess_input(image)
        self.input_dict = {self.input_name: p_frame}
        self.input = image
        result = self.net.infer(self.input_dict)
        coords = self.preprocess_output(result, image)
        image = self.draw_outputs(image, coords)
        # left_eye, right_eye, coords, input_s = self.preprocess_output(result)
        return coords, image
        # return left_eye, right_eye, coords, input_s

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.

        Name: "data" , shape: [1x3x48x48] - An input image in the format [BxCxHxW], where:

        B - batch size
        C - number of channels
        H - image height
        W - image width

        The expected color order is BGR.
        '''
        image = cv2.resize(
            image, (self.input_shape[3], self.input_shape[2]), interpolation=cv2.INTER_AREA)
        p_image = image.reshape(
            1, 3, self.input_shape[2], self.input_shape[3])

        return p_image, image
    # def preprocess_output(self, outputs):
    #     '''
    #     Before feeding the output of this model to the next model,
    #     you might have to preprocess the output. This function is where you can do that.
    #     '''
    #     self.input = self.input.transpose(1, 2, 0)
    #     print(f"input shape {self.input.shape[0]}")
    #     print(f"input shape {outputs[0][1][0][0]}")
    #     leftEye = self.input[int(self.input.shape[0]*outputs[0][1][0][0])-5:int(self.input.shape[0]*outputs[0][1][0][0])+5,
    #                          int(self.input.shape[1]*outputs[0][0][0][0])-5:int(self.input.shape[1]*outputs[0][0][0][0])+5]
    #     rightEye = self.input[int(self.input.shape[0]*outputs[0][3][0][0])-5:int(self.input.shape[0]*outputs[0][3][0][0])+5,
    #                           int(self.input.shape[1]*outputs[0][2][0][0])-5:int(self.input.shape[1]*outputs[0][2][0][0])+5]
    #     if leftEye.any():
    #         leftEye = cv2.resize(leftEye, (60, 60)).transpose((2, 0, 1))
    #         leftEye = leftEye.reshape(1, *leftEye.shape)
    #     if rightEye.any():
    #         rightEye = cv2.resize(rightEye, (60, 60)).transpose((2, 0, 1))
    #         rightEye = rightEye.reshape(1, *rightEye.shape)
    #     return leftEye, rightEye, ((outputs[0][0], outputs[0][1]), (outputs[0][2], outputs[0][3])), self.input

    def preprocess_output(self, outputs, image):
    #     '''
    #     Before feeding the output of this model to the next model,
    #     you might have to preprocess the output. This function is where you can do that.

    #     The net outputs a blob with the shape: [1, 10], containing a row-vector of 10 floating point values for five landmarks coordinates in the form (x0, y0, x1, y1, ..., x5, y5). All the coordinates are normalized to be in range [0,1].
    #     '''

        # print("preprocess output")
        height, width = image.shape[0:2]
        print(f"Output Height: {height} :: Width: {width}")
        coordinates = []
        radius = 30
        results = outputs['95'].flatten()
        for i in range(0,len(results),2):
            point = (int(results[i]*width), int(results[i+1]*height))
            xmin = point[0]-radius
            ymin = point[1]-radius
            xmax = point[0]+radius
            ymax = point[1]+radius
            coordinates.append((xmin, ymin, xmax, ymax))
        return coordinates

    def draw_outputs(self, image, coords):
        '''
        TODO: This method needs to be completed by you
        '''
        # print("Draw output")
        print(coords)
        height, width = image.shape[0:2]
        print(f"Draw Height: {height} :: Width: {width}")
        for box in coords:
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 1)
            #  cv2.circle(image, point, 3, (0, 0, 255), 1)
        return image


def main():
    model = 'models/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009'
    input_image = 'bin/cropped_face.png'

    fld = FacialLandmarkDetector(model)
    fld.load_model()

    face = cv2.imread(input_image)

    coords, landmark_detection = fld.predict(face)
    cv2.imshow('Landmark Detection', landmark_detection)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ =='__main__':

    main()