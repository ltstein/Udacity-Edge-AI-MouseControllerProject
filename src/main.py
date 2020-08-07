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

    if not os.path.exists('output'):
        os.makedirs('output')
    print(f"Logging to: output/{log_name}")
    log = open('output/'+log_name, 'w+')

    print("Initializing models...")

    
    fd = FaceDetector(
        model_name='models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001',
        device=args.device,
        extensions=None)
    
    fd.load_model()

    if args.v: print(f"Face Detection Load Time: {fd.load_time}")
    
    
    hpe = HeadPoseEstimator(
        model_name=f'models/intel/head-pose-estimation-adas-0001/{args.hpe}/head-pose-estimation-adas-0001',
        device=args.device,
        extensions=None)
    hpe.load_model()
    
    if args.v: print(f"Head Pose Estimation Load Time: {hpe.load_time}")
    

    
    fld = FacialLandmarkDetector(
        model_name=f'models/intel/landmarks-regression-retail-0009/{args.fld}/landmarks-regression-retail-0009',
        device=args.device,
        extensions=None)
    fld.load_model()
    
    if args.v: print(f"Facial Landmarks Detection Load Time: {fld.load_time}")
    

    
    ge = GazeEstimator(
        model_name=f'models/intel/gaze-estimation-adas-0002/{args.ge}/gaze-estimation-adas-0002',
        device=args.device,
        extensions=None)
    ge.load_model()
    
    if args.v: print(f"Gaze Estimation Load Time: {ge.load_time}")

    image = False

    print("Initializing source feed...")
    feed=InputFeeder(input_type=args.input_type, input_file=args.input_file)
    if args.input_type ==  'image':
        image = True

    feed.load_data()

    for batch in feed.next_batch():
        if args.v:
            print()
        cv2.imshow('Batch', batch)
        if image:
            cv2.imwrite('output/Batch.png', batch)


        coords, bounding_face = fd.predict(batch)
        if not coords:
            print("No face")
            continue
        if image: cv2.imwrite('output/Face.png', bounding_face)
        box = coords[0]
        face = bounding_face[box[1]:box[3], box[0]:box[2]]

        if args.v:
            print(f"Face Time: {fd.infer_time}")
        log.write("FD_infer: " + str(fd.infer_time) + "\n")
        if image:
            cv2.imshow('Cropped Face', face)


        # Landmark Detection
        coords, landmark_detection, landmark_points = fld.predict(face)
        if image: cv2.imwrite('output/Landmarks.png', landmark_detection)
        if image: cv2.imshow('Landmark Detection', landmark_detection)
        if args.v: print(f"Landmark Time: {fld.infer_time}")
        log.write("FLD_infer: " + str(fld.infer_time) + "\n")
        right_box, left_box = coords[0:2]
        if args.v: print(f"Eye Coords: {coords}")

        if left_box == None or right_box == None:
            print("No eyes")
            continue

        left_eye = face[left_box[1]:left_box[3], left_box[0]:left_box[2]]
        cv2.putText(face, 'L', (left_box[0], left_box[3]),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        right_eye = face[right_box[1]:right_box[3], right_box[0]:right_box[2]]
        cv2.putText(face, 'R', (right_box[0], right_box[3]),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        if args.v: 
            print(f"Eye Shape: {left_eye.shape} :: {right_eye.shape}")

        #Head Pose Estimation
        head_yaw, head_pitch, head_roll = hpe.predict(face)
        if args.v: print(f"Head Pose Time: {hpe.infer_time}")
        log.write("HPE_infer: " + str(hpe.infer_time) + "\n")
        head_angles = [head_yaw[0][0], head_pitch[0][0], head_roll[0][0]]


        #Gaze Estimation
        # expects pose as  (yaw, pitch, and roll) 
        gaze = ge.predict(left_eye, right_eye, head_angles)

        if args.v:
            print(f"Gaze Time: {ge.infer_time}")
        log.write("GE_infer: " + str(ge.infer_time) + "\n")
        gaze_point = (int(gaze[0][0]*50), int(gaze[0][1]*50))

        arrows = cv2.arrowedLine(face, landmark_points[0], (
            landmark_points[0][0] + gaze_point[0], landmark_points[0][1] - gaze_point[1]), (0, 0, 255), 2)
        arrows = cv2.arrowedLine(face, landmark_points[1], (
            landmark_points[1][0] + gaze_point[0], landmark_points[1][1] - gaze_point[1]), (0, 0, 255), 2)
        if image:
            cv2.imwrite('output/Gaze.png', arrows)

        if not image:
            mouse = MouseController(precision='medium', speed='medium')
            mouse.move(gaze[0][0],gaze[0][1])
        
        if image: 
            cv2.imshow('Arrows', arrows)

        if image: 
            log.write("FD_LoadTime: " + str(fd.load_time) + "\n")
            log.write("FD_PreprocessTime: " + str(fd.preprocess_input_time) + "\n")
            log.write("FD_PostrocessTime: " + str(fd.preprocess_output_time) + "\n")

            log.write("FLD_LoadTime: " + str(fld.load_time) + "\n")
            log.write("FLD_PreprocessTime: " + str(fld.preprocess_input_time) + "\n")
            log.write("FLD_PostprocessTime: " + str(fld.preprocess_output_time) + "\n")

            log.write("HPE_LoadTime: " + str(hpe.load_time) + "\n")
            log.write("HPE_PreprocessTime: " + str(hpe.preprocess_input_time) + "\n")
            
            log.write("GE_LoadTime: " + str(ge.load_time) + "\n")
            log.write("GE_PreprocessTime: " + str(ge.preprocess_input_time) + "\n")

            cv2.waitKey(0)
        else:
            if cv2.waitKey(15) & 0xFF == ord('q'):
                break

    feed.close()
    log.close()
    cv2.destroyAllWindows

if __name__ =='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--device', default='CPU') 
    parser.add_argument('-hpe', default='FP32', choices=[
                        'FP16', 'FP32', 'FP32-INT8'], type=str, help='Set precision for Head Pose Estimation Model')
    parser.add_argument('-fld', default='FP32', choices=[
                        'FP16', 'FP32', 'FP32-INT8'], type=str, help='Set precision for Facial Landmark Detection Model')
    parser.add_argument('-ge', default='FP32', choices=[
                        'FP16', 'FP32', 'FP32-INT8'], type=str, help='Set precision for Gaze Estimation Model')
    parser.add_argument('-it', '--input_type', default='image', type=str,
                        choices=['video', 'cam', 'image']) 
    parser.add_argument('-if', '--input_file',
                        default='bin/demo_1.png', type=str, help='Set path if using input file') 
    parser.add_argument('-v', action='store_true',
                        help='Increase verbosity of console output')

    args = parser.parse_args()

    main(args)
