import cv2
import threading
import queue
import numpy as np
import time
from tflite_runtime.interpreter import Interpreter
import cProfile


# Initialize TensorFlow Lite interpreter
modelpath = 'detect.tflite'
lblpath = 'labelmap.txt'
min_conf = 0.5
interpreter = Interpreter(model_path=modelpath)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Read labels
with open(lblpath, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

def capture_frames(cap, frame_queue):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_queue.qsize() < 10:  # Limit the size of the queue
            frame_queue.put(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def determine_position(xmin, xmax, ymin, ymax, imW, imH):
    x_center = (xmin + xmax) / 2
    y_center = (ymin + ymax) / 2

    horizontal_position = "Centre"
    if x_center < imW / 3:
        horizontal_position = "Left"
    elif x_center > 2 * imW / 3:
        horizontal_position = "Right"

    vertical_position = "Centre"
    if y_center < imH / 3:
        vertical_position = "Top"
    elif y_center > 2 * imH / 3:
        vertical_position = "Bottom"

    return f"{vertical_position} {horizontal_position}"

def process_frame(frame, start_time, frame_count, fps):
    imH, imW, _ = frame.shape
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (300, 300))
    input_data = np.expand_dims(image_resized, axis=0)

    # Normalize pixel values if using a floating model
    if input_details[0]['dtype'] == np.float32:
        input_data = (np.float32(input_data) - 127.5) / 127.5

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[1]['index'])[0]
    classes = interpreter.get_tensor(output_details[3]['index'])[0]
    scores = interpreter.get_tensor(output_details[0]['index'])[0]

    # Process detections
    for i in range(len(scores)):
        if scores[i] > min_conf and scores[i] <= 1.0:
            ymin, xmin, ymax, xmax = boxes[i]
            ymin, xmin, ymax, xmax = int(ymin * imH), int(xmin * imW), int(ymax * imH), int(xmax * imW)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
            object_name = labels[int(classes[i])]
            label = '%s: %d%%' % (object_name, int(scores[i] * 100))
            position = determine_position(xmin, xmax, ymin, ymax, imW, imH)
            cv2.putText(frame, position, (xmin, ymin-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, label, (xmin, ymin - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Calculate FPS
    frame_count += 1
    if (time.time() - start_time) > 1:
        fps = frame_count / (time.time() - start_time)
        frame_count = 0
        start_time = time.time()
    fps_text = 'FPS: {:.2f}'.format(fps)
    cv2.putText(frame, fps_text, (frame.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame, start_time, frame_count, fps

def process_frames(frame_queue, display_queue):
    start_time = time.time()
    frame_count = 0
    fps = 0
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            processed_frame, start_time, frame_count, fps = process_frame(frame, start_time, frame_count, fps)
            display_queue.put(processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

def display_frames(display_queue):
    while True:
        if not display_queue.empty():
            frame = display_queue.get()
            cv2.imshow('Processed Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

def main():
    frame_queue = queue.Queue()
    display_queue = queue.Queue()
    cap = cv2.VideoCapture(0)

    thread_capture = threading.Thread(target=capture_frames, args=(cap, frame_queue))
    thread_process = threading.Thread(target=process_frames, args=(frame_queue, display_queue))
    thread_display = threading.Thread(target=display_frames, args=(display_queue,))

    thread_capture.start()
    thread_process.start()
    thread_display.start()

    thread_capture.join()
    thread_process.join()
    thread_display.join()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    cProfile.run('main()')
    
