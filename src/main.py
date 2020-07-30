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
from mouse_controller import MouseController

# import model


def main(args):
    print("Main script running...")
    log_name = 'stats_' + args.device + '_' + args.hpe + args.fld + args.ge
    print(f"Logging to: {log_name}")
    log = open(log_name, 'w+')

    print("Initializing models...")

    start_fd_load_time = time.time()
    
    fd = FaceDetector(
        model_name='models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001',
        device=args.device,
        extensions=None)
    fd.load_model()
    fd_load_time = time.time() - start_fd_load_time

    if args.v: print(f"Face Detection Load Time: {fd_load_time}")
    log.write("FD_LD: " + str(fd_load_time) + "\n")
    
    start_hpe_load_time = time.time()
    hpe = HeadPoseEstimator(
        model_name=f'models/intel/head-pose-estimation-adas-0001/{args.hpe}/head-pose-estimation-adas-0001',
        device=args.device,
        extensions=None)
    hpe.load_model()
    hpe_load_time = time.time() - start_hpe_load_time
    if args.v: print(f"Head Pose Estimation Load Time: {hpe_load_time}")
    log.write("HPE_LD: " + str(hpe_load_time) + "\n")

    start_fld_load_time = time.time()
    fld = FacialLandmarkDetector(
        model_name=f'models/intel/landmarks-regression-retail-0009/{args.fld}/landmarks-regression-retail-0009',
        device=args.device,
        extensions=None)
    fld.load_model()
    fld_load_time = time.time() - start_fld_load_time
    if args.v: print(f"Facial Landmarks Detection Load Time: {fld_load_time}")
    log.write("FLD_LD: " + str(fld_load_time) + "\n")

    start_ge_load_time = time.time()
    ge = GazeEstimator(
        model_name=f'models/intel/gaze-estimation-adas-0002/{args.ge}/gaze-estimation-adas-0002',
        device=args.device,
        extensions=None)
    ge.load_model()
    ge_load_time = time.time() - start_ge_load_time
    if args.v: print(f"Gaze Estimation Load Time: {ge_load_time}")
    log.write("GE_LD: " + str(ge_load_time) + "\n")

    

    print("Initializing source feed...")
    feed=InputFeeder(input_type=args.input_type, input_file=args.input_file)
    if args.input_type ==  'image':
        image = True
    else:
        image = False
    # feed=InputFeeder(input_type='video', input_file='bin/demo.mp4')
    # feed=InputFeeder(input_type='image', input_file='bin/demo_1.png')
    # feed=InputFeeder(input_type='cam')
    feed.load_data()

    for batch in feed.next_batch():
        if args.v: print()
        cv2.imshow('Batch',batch)
        if image: cv2.imwrite('Batch.png', batch)
        # Press Q on keyboard to  exit 
        # if cv2.waitKey(5) & 0xFF == ord('q'):
        #     break

        coords, bounding_face = fd.predict(batch)
        # print(f"coords: {coords}")
        if not coords:
            print("No face")
            continue
        if image: cv2.imwrite('Face.png', bounding_face)
        box = coords[0]
        face = bounding_face[box[1]:box[3], box[0]:box[2]]
        # print(f"Face Dim Height: {face.shape[0]} :: Width: {face.shape[1]}")
        if args.v: print(f"Face Time: {fd.infer_time}")
        log.write("FD_infer: " + str(fd.infer_time) + "\n")
        if image: cv2.imshow('Cropped Face', face)
        # cv2.imshow('Face Detection', bounding_face)
        
        #Landmark Detection
        coords, landmark_detection, landmark_points = fld.predict(face)
        cv2.imwrite('Landmarks.png', landmark_detection)
        if image: cv2.imshow('Landmark Detection', landmark_detection)
        if args.v: print(f"Landmark Time: {fld.infer_time}")
        log.write("FLD_infer: " + str(fld.infer_time) + "\n")
        right_box, left_box = coords[0:2]
        if args.v: print(f"Eye Coords: {coords}")

        if left_box == None or right_box == None:
            print("No eyes")
            continue

        # cv2.putText(image, text, (x, y), font, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])

        left_eye = face[left_box[1]:left_box[3], left_box[0]:left_box[2]]
        cv2.putText(face, 'L', (left_box[0], left_box[3]),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        right_eye = face[right_box[1]:right_box[3], right_box[0]:right_box[2]]
        cv2.putText(face, 'R', (right_box[0], right_box[3]),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        if args.v: print(f"Eye Shape: {left_eye.shape} :: {right_eye.shape}")

        #Head Pose Estimation
        head_yaw, head_pitch, head_roll = hpe.predict(face)
        if args.v: print(f"Head Pose Time: {hpe.infer_time}")
        log.write("HPE_infer: " + str(hpe.infer_time) + "\n")
        head_angles = [head_yaw[0][0], head_pitch[0][0], head_roll[0][0]]


        #Gaze Estimation
        # expects pose as  (yaw, pitch, and roll) 
        gaze = ge.predict(left_eye, right_eye, head_angles)
        # print(f"Gaze: {gaze}")
        # image, start_point, end_point, color[, thickness
        # print(f"Points: {landmark_points}")
        if args.v: print(f"Gaze Time: {ge.infer_time}")
        log.write("GE_infer: " + str(ge.infer_time) + "\n")
        gaze_point = (int(gaze[0][0]*50), int(gaze[0][1]*50))
        # print(f"Gaze point: {gaze_point}")

        # cv2.arrowedLine(image, start_point, end_point, color[, thickness[, line_type[, shift[, tipLength]]]])
        arrows = cv2.arrowedLine(face, landmark_points[0], (landmark_points[0][0] + gaze_point[0], landmark_points[0][1] - gaze_point[1]), (0,0,255), 2)
        arrows = cv2.arrowedLine(face, landmark_points[1], (landmark_points[1][0] + gaze_point[0], landmark_points[1][1] - gaze_point[1]), (0,0,255), 2)
        if image: cv2.imwrite('Gaze.png', arrows)
        
        if not image: 
            mouse = MouseController(precision='medium', speed='medium')
            mouse.move(gaze[0][0],gaze[0][1])
        
        if image: cv2.imshow('Arrows', arrows)

        if image: 
            cv2.waitKey(0)
        else:
            if cv2.waitKey(15) & 0xFF == ord('q'):
                break

    feed.close()
    log.close()
    cv2.destroyAllWindows

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model', required=True)
    parser.add_argument('--device', default='CPU') #CPU, iGPU
    parser.add_argument('--hpe', default='FP32') #FP16, FP32, FP32-INT8
    parser.add_argument('--fld', default='FP32') #FP16, FP32, FP32-INT8
    parser.add_argument('--ge', default='FP32')  #FP16, FP32, FP32-INT8
    parser.add_argument('--input_type', default='image') #video, cam, image
    parser.add_argument('--input_file', default='bin/demo_1.png') #path to file
    parser.add_argument('--v', action='store_true')

    args = parser.parse_args()

    main(args)
