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

    start_ge_load_time = time.time()
    ge = GazeEstimator(
        model_name='models/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002',
        device='CPU',
        extensions=None)
    ge.load_model()
    ge_load_time = time.time() - start_ge_load_time
    


    print("Initializing source feed...")
    # feed=InputFeeder(input_type='video', input_file='bin/demo.mp4')
    feed=InputFeeder(input_type='image', input_file='bin/demo.png')
    # feed=InputFeeder(input_type='cam')
    feed.load_data()
    
    for batch in feed.next_batch():
        # cv2.imshow('Frame',batch)
        # Press Q on keyboard to  exit 
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
        coords, bounding_face = fd.predict(batch)
        box = coords[0]
        face = bounding_face[box[1]:box[3], box[0]:box[2]]
        # print(f"Face Dim Height: {face.shape[0]} :: Width: {face.shape[1]}")
        cv2.imshow('Cropped Face', face)
        cv2.imshow('Face Detection', bounding_face)
        
        #Landmark Detection
        coords, landmark_detection = fld.predict(face)
        cv2.imshow('Landmark Detection', landmark_detection)
        left_box, right_box = coords[0:2]
        # print(f"Left box: {left_box}")
        right_eye = face[left_box[1]:left_box[3], left_box[0]:left_box[2]]
        left_eye = face[right_box[1]:right_box[3], right_box[0]:right_box[2]]
        # cv2.imshow('left_eye', left_eye)
        # cv2.imshow('right_eye', right_eye)

        # left_eye, right_eye, eyecoord, previewCanvas = fld.predict(face)


        # x0 = int(max(0, face.shape[1]*eyecoord[0][0][0]-5))
        # y0 = int(max(0, face.shape[0]*eyecoord[0][1][0]-5))
        # x1 = int(face.shape[1]*eyecoord[0][0][0]+5)
        # y1 = int(face.shape[0]*eyecoord[0][1][0]+5)
        # previewCanvas = cv2.rectangle(face, (x0, y0), (x1, y1), (255,255,255))
                
        # previewCanvas = cv2.rectangle(previewCanvas, (int(max(0, face.shape[1]*eyecoord[1][0][0]-5)), int(max(0, face.shape[0]*eyecoord[1][1][0]-5))), (int(face.shape[1]*eyecoord[1][0][0]+5), int(face.shape[0]*eyecoord[1][1][0]+5)), (255,255,255))

        # cv2.imshow('preview', previewCanvas)

        #Head Pose Estimation
        coords, head_pose = hpe.predict(face)
        # head_angles = [head_pose['angle_y_fc'], head_pose['angle_p_fc'], head_pose['angle_r_fc']]

        #Gaze Estimation
        # expects pose as  (yaw, pitch, and roll) 
        # gaze = ge.predict(left_eye, right_eye,head_angles)


        cv2.waitKey(0)
    feed.close()
    cv2.destroyAllWindows

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
