#main.py

import sys
import argparse
import cv2
import os
from openvino.inference_engine import IENetwork, IECore
import time
import numpy as np

#Import local scripts for processing pipeline
from face_detection import FaceDetector
from facial_landmarks_detection import FacialLandmarkDetector
from head_pose_estimation import HeadPoseEstimator
from gaze_estimation import GazeEstimator
from input_feeder import InputFeeder
# import model


def main(args):
    print("Main script running...")

    print("Initializing models...")

    start_fd_load_time = time.time()
    
    fd = FaceDetector(
        model_name='models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001',
        device='CPU',
        extensions=None)
    fd.load_model()
    fd_load_time = time.time() - start_fd_load_time

    # print(f"FD input: {fd.input_shape}")

    
    start_hpe_load_time = time.time()
    hpe = HeadPoseEstimator(
        model_name='models/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001',
        device='CPU',
        extensions=None)
    hpe.load_model()
    hpe_load_time = time.time() - start_hpe_load_time

    start_fld_load_time = time.time()
    fld = FacialLandmarkDetector(
        model_name='models/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009',
        device='CPU',
        extensions=None)
    fld.load_model()
    fld_load_time = time.time() - start_fld_load_time

    # start_ge_load_time = time.time()
    # ge = GazeEstimator(
    #     model_name='models/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002',
    #     device='CPU',
    #     extensions=None)
    # ge.load_model()
    # ge_load_time = time.time() - start_ge_load_time
    


    print("Initializing source feed...")
    # feed=InputFeeder(input_type='video', input_file='bin/demo.mp4')
    feed=InputFeeder(input_type='image', input_file='bin/demo.png')
    # feed=InputFeeder(input_type='cam')
    feed.load_data()
    
    for batch in feed.next_batch():
        # cv2.imshow('Frame',batch)
        # Press Q on keyboard to  exit 
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
        coords, bounding_face = fd.predict(batch)
        box = coords[0]
        face = bounding_face[box[1]:box[3], box[0]:box[2]]
        print(f"Face Dim Height: {face.shape[0]} :: Width: {face.shape[1]}")
        cv2.imshow('Cropped Face', face)
        cv2.imshow('Face Detection', bounding_face)
        
        #Landmark Detection
        coords, landmark_detection = fld.predict(face)
        cv2.imshow('Landmark Detection', landmark_detection)

        #Head Pose Estimation
        coords, head_pose = hpe.predict(face)


        cv2.waitKey(0)
    feed.close()


if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model', required=True)
    # parser.add_argument('--device', default='CPU')
    # parser.add_argument('--video', default=None)
    # parser.add_argument('--queue_param', default=None)
    # parser.add_argument('--output_path', default='/results')
    # parser.add_argument('--threshold', default=0.60)

    args = parser.parse_args()

    main(args)
