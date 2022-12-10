import time
import torch
import cv2
import requests
from bs4 import BeautifulSoup
import json

camera_num = 1
# Model
model_l = torch.hub.load('ultralytics/yolov5', 'yolov5n')  # or yolov5n - yolov5x6, custom


def get_motion():
    try:
        r = requests.get('http://192.168.0.112:80', timeout=3)
        soup = BeautifulSoup(r.content, 'lxml')
        p = soup.p.text
        return p
    except Exception as e:
        print(e)
        return 0


# Inference
def nn(img, model):
    results = model(img)
    # Results
    # print(results)

    a = results.pandas().xyxy[0].name

    print(str(a).count('person'))
    results.show()
    return str(a).count('person')


def get_cam_info():
    cam = cv2.VideoCapture(camera_num)
    ret, frame = cam.read()
    img_name = "opencv_frame.png"
    cv2.imwrite(img_name, frame)
    cam.release()
    return nn("opencv_frame.png", model_l)


while True:
    try:
        m = get_motion()
        if m == '1':
            get_cam_info()
            time.sleep(0.2)
        else:
            print(m)
            time.sleep(0.2)
    except Exception as e:
        print(e)