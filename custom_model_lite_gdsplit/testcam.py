import cv2
import time

cap = cv2.VideoCapture(0)  # 0 - video input from webcam

# Initialize FPS calculation variables
frame_count = 0
start_time = time.time()
fps_display_interval = 1  # seconds
fps = 0

while(True):
    ret, frame = cap.read()
    if not ret:
        break

    # Increment frame count
    frame_count += 1

    # Calculate FPS at an interval of 'fps_display_interval'
    if (time.time() - start_time) > fps_display_interval:
        fps = frame_count / (time.time() - start_time)
        frame_count = 0
        start_time = time.time()

    # Display FPS on the top right corner of the frame
    fps_text = 'FPS: {:.2f}'.format(fps)
    cv2.putText(frame, fps_text, (frame.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('output', frame)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break

cap.release()
cv2.destroyAllWindows()
