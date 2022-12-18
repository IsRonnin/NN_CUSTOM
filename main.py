import threading
import time
import torch
import cv2
import requests
from bs4 import BeautifulSoup
import json
from PIL import Image
from io import BytesIO
c = 0
camera_num = 0
model_l = 'Yolov7_custom.pt'  # or yolov5n - yolov5x6, custom
url = 'https://images.squarespace-cdn.com/content/v1/6102a0b517117760b1defa27/1628170822958-WMVE9EU7QJ3COZV5I5ZE/7f18a2d9-ab9f-4832-b9cd-a4952a036404.jpg?format=500w'
response = requests.get(url)
img = Image.open(BytesIO(response.content))
def get_motion():
    try:
        r = requests.get('http://192.168.0.112:80', timeout=3)
        soup = BeautifulSoup(r.content, 'lxml')
        p = soup.p.text
        return p
    except Exception as e:
        print(e)
        return 0

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

def counter():
    global c
    while True:
        time.sleep(1)
        c += 1
        print(c)


t = threading.Thread(target=counter, args=())
t.start()
print('Ready!')
while True:
    try:
        #m = get_motion()
        if input() == '':
            print('work!')
            #get_cam_info()
            nn(img, model_l)
            time.sleep(0.2)
        else:
            time.sleep(0.2)
    except Exception as e:
        print(e)