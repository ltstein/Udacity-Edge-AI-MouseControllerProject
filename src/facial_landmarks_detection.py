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
        # print("Initializing Facial Landmarks Model...")
        self.model_weights = model_name+'.bin'
        self.model_structure = model_name+'.xml'
        self.device = device

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
        self.input_blob, self.resized_input_image = self.preprocess_input(image)
        input_dict = {self.input_name: self.input_blob}
        self.input = image
        start_time = time.time()
        self.net.start_async( request_id = 0, inputs=input_dict)
        status = self.net.requests[0].wait(-1)

        if status == 0:
            output = self.net.requests[0].outputs[self.output_name]
            self.infer_time = time.time() - start_time
        
        if __name__ == "facial_landmarks_detection":
            coords, points = self.preprocess_output(output.flatten(), self.input.shape[0], self.input.shape[1])
            image = self.draw_outputs(image, coords)
            return coords, image, points
        # return left_eye, right_eye, coords_debug, input_s
        if __name__ == "__main__":
            return infer_result, output

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
        resized_input_image = cv2.resize(
            image, (self.input_shape[3], self.input_shape[2]), interpolation=cv2.INTER_AREA)

        resized_input_image = resized_input_image.transpose((2,0,1))

        input_blob = resized_input_image.reshape(
            1, 3, self.input_shape[2], self.input_shape[3])


        return input_blob, resized_input_image


    def preprocess_output(self, outputs, height, width):
    #     '''
    #     Before feeding the output of this model to the next model,
    #     you might have to preprocess the output. This function is where you can do that.

    #     The net outputs a blob with the shape: [1, 10], containing a row-vector of 10 floating point values for five landmarks coordinates in the form (x0, y0, x1, y1, ..., x5, y5). All the coordinates are normalized to be in range [0,1].
    #     '''

        # print("preprocess output")
        coordinates = []
        points = []
        radius = 30
        height_factor = height/self.input_shape[0]
        width_factor = width/self.input_shape[1]

        # print(f"Outputs: {outputs[0:2]}")
        # print(f"Preprocess Outputs: {outputs}")
        for i in range(0,len(outputs),2):
            point = (int((outputs[i]*self.input_shape[1])*width_factor), int((outputs[i+1]*self.input_shape[0])*height_factor))
            xmin = max(min(width, point[0]-radius), 0) 
            ymin = max(min(height, point[1]-radius), 0) 
            xmax = max(min(width, point[0]+radius), 0) 
            ymax = max(min(height, point[1]+radius), 0) 
            # https://stackoverflow.com/questions/5996881/how-to-limit-a-number-to-be-within-a-specified-range-python
            #Sanitize coordinates to be between 0 and height or width
            coordinates.append((xmin, ymin, xmax, ymax))
            points.append(point)
        return coordinates, points

    def draw_outputs(self, image, coords):
        '''
        TODO: This method needs to be completed by you
        '''
        drawn_image = image
        # print(f"Draw coords: {coords}")
        for box in coords:
            cv2.rectangle(drawn_image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 1)
        return drawn_image


def main():
    model = 'models/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009'
    input_image = 'bin/cropped_face.png'

    fld = FacialLandmarkDetector(model)
    fld.load_model()


    face = cv2.imread(input_image)

    infer_result, output = fld.predict(face)

    cropped_face_coords = fld.preprocess_output(infer_result['95'].flatten(), face.shape[0], face.shape[1])
    cropped_face_drawn = fld.draw_outputs(face, cropped_face_coords)
    cv2.imshow('Cropped Face Drawn', cropped_face_drawn)

    cropped_face_coords_output = fld.preprocess_output(output.flatten(), face.shape[0], face.shape[1])
    cropped_face_drawn_output = fld.draw_outputs(face, cropped_face_coords_output)
    cv2.imshow('O Cropped Face Drawn', cropped_face_drawn_output)


    # cv2.imshow('Landmark Detection', landmark_detection)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ =='__main__':

    main()