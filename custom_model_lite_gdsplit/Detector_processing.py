import cv2
import multiprocessing
import numpy as np
import time
import cProfile
from tflite_runtime.interpreter import Interpreter

modelpath = 'detect.tflite'
lblpath = 'labelmap.txt'
min_conf = 0.5

def load_labels(label_path):
    with open(label_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def capture_frames(cap, frame_queue, fps_queue):
    frame_count = 0
    start_time = time.time()
    fps = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time >= 1:
            fps = frame_count / elapsed_time
            fps_queue.put(fps)
            frame_count = 0
            start_time = time.time()

        if frame_queue.qsize() < 10:
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

def process_frame(frame, interpreter, labels):
    imH, imW, _ = frame.shape
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (300, 300))
    input_data = np.expand_dims(image_resized, axis=0)
    input_data = (np.float32(input_data) - 127.5) / 127.5

    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
    interpreter.invoke()

    boxes = interpreter.get_tensor(interpreter.get_output_details()[1]['index'])[0]
    classes = interpreter.get_tensor(interpreter.get_output_details()[3]['index'])[0]
    scores = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])[0]

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

    return frame

def process_frames(frame_queue, display_queue, fps_queue):
    interpreter = Interpreter(model_path=modelpath)
    interpreter.allocate_tensors()
    labels = load_labels(lblpath)

    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            processed_frame = process_frame(frame, interpreter, labels)
            if not fps_queue.empty():
                fps = fps_queue.get()
                cv2.putText(processed_frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            display_queue.put(processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

def display_frames(display_queue):
    last_time = time.time()
    frame_count = 0
    while True:
        if not display_queue.empty():
            frame = display_queue.get()
            frame_count += 1

            # Calculate FPS
            current_time = time.time()
            if (current_time - last_time) >= 1:
                fps = frame_count / (current_time - last_time)
                frame_count = 0
                last_time = current_time
                fps_text = f'FPS: {fps:.2f}'
                cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Processed Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


def main():
    frame_queue = multiprocessing.Queue()
    display_queue = multiprocessing.Queue()
    fps_queue = multiprocessing.Queue()
    cap = cv2.VideoCapture(0)

    process_capture = multiprocessing.Process(target=capture_frames, args=(cap, frame_queue, fps_queue))
    process_process = multiprocessing.Process(target=process_frames, args=(frame_queue, display_queue, fps_queue))
    process_display = multiprocessing.Process(target=display_frames, args=(display_queue,))

    process_capture.start()
    process_process.start()
    process_display.start()

    process_capture.join()
    process_process.join()
    process_display.join()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    cProfile.run('main()')
