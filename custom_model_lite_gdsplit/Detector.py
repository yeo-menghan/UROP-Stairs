import numpy as np
import cv2
import os
import sys
import glob
import random
import importlib.util
import time
# from tensorflow.lite.python.interpreter import Interpreter
from tflite_runtime.interpreter import Interpreter
# import tflite_runtime.interpreter as Interpreter

import matplotlib
import matplotlib.pyplot as plt

modelpath='detect.tflite'
lblpath='labelmap.txt'
min_conf=0.5
cap = cv2.VideoCapture(0) # 0 - video input from webcam

interpreter = Interpreter(model_path=modelpath)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

float_input = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Define new size for resizing the input images
new_width, new_height = 300, 300


with open(lblpath, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Initialize FPS calculation variables
frame_count = 0
start_time = time.time()
fps_display_interval = 1  # seconds
fps = 0
# last_display_time = time.time()
# display_interval = 0.1  # Update the display every 0.1 seconds


while(True):
    ret, frame =cap.read()
    if not ret:
        break
    # Increment frame count
    frame_count += 1

    # Calculate FPS at an interval of 'fps_display_interval'
    if (time.time() - start_time) > fps_display_interval:
        fps = frame_count / (time.time() - start_time)
        frame_count = 0
        start_time = time.time()

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imH, imW, _ = frame.shape
    image_resized = cv2.resize(image_rgb, (new_width, new_height))
    input_data = np.expand_dims(image_resized, axis=0)
    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if float_input:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[1]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[3]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[0]['index'])[0] # Confidence of detected objects

    detections = []


    for i in range(len(scores)):
        if ((scores[i] > min_conf) and (scores[i] <= 1.0)):
            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
            # Draw label
            object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
            label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
            detections.append([object_name, scores[i], xmin, ymin, xmax, ymax])

            #### Determine position on screen ####

            # Calculate the centre of the bounding box
            x_centre = (xmin + xmax) / 2
            y_centre = (ymin + ymax) / 2

            # Determine the horizontal position
            if x_centre < imW / 3:
                horizontal_position = "Left"
            elif x_centre > 2 * imW / 3:
                horizontal_position = "Right"
            else:
                horizontal_position = "Centre"

            # Determine the vertical position
            if y_centre < imH / 3:
                vertical_position = "Top"
            elif y_centre > 2 * imH / 3:
                vertical_position = "Bottom"
            else:
                vertical_position = "Centre"

            # Combine horizontal and vertical position
            position = f"{vertical_position} {horizontal_position}"

            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

            # Display the position
            cv2.putText(frame, position, (xmin, ymin-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            print(f"Current time: {current_time}, Label: {label}, Position: {position}")

    # Display FPS on the top right corner of the frame
    fps_text = 'FPS: {:.2f}'.format(fps)
    cv2.putText(frame, fps_text, (frame.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # # Display the frame at set intervals
    # if time.time() - last_display_time > display_interval:
    #     cv2.imshow('output', frame)
    #     last_display_time = time.time()

    cv2.imshow('output',frame)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break

cap.release()
cv2.destroyAllWindows()
