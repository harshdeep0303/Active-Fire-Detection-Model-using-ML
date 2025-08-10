from ultralytics import YOLO
import cv2
import cvzone
import math
import time
from playsound import playsound

model = YOLO('fire.pt')

classNames = [ 'fire', 'smoke']

# import serial
# esp32 = serial.Serial('COM6', 9600)
prev_frame_time = 0
new_frame_time = 0
cap = cv2.VideoCapture(0)

while True:
    new_frame_time = time.time()
    success, img = cap.read()
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            conf = math.ceil((box.conf[0] * 100))
            cls = int(box.cls[0])

            cvzone.cornerRect(img, (x1, y1, w, h))
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(30, y1)), scale=1, thickness=1)

            # if classNames[cls] == 'fire':
            #     playsound('school-fire-alarm-loud-beepflac-14807.mp3')
    
            # elif classNames[cls] == 'smoke' :
            #     esp32.write(b's')
            # else:
            #     esp32.write(b'n')
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(fps)

    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()