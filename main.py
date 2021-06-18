import cv2

import tkinter as tk
from tkinter import *
from tkinter import filedialog


def clicked():
   path=filedialog.askopenfilename(filetypes=[("Image File",'.jpg')])
   print(path)
   imageread(path)


def imageread(kk):
    img = cv2.imread(kk)
    #classNames = ["person"]
    classNames = []




    classFile = 'coco.names'
    with open(classFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')

    configpath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weigthpath = 'frozen_inference_graph.pb'

    net = cv2.dnn_DetectionModel(weigthpath, configpath)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)
    classIds, confs, bbox = net.detect(img, confThreshold=0.55)
    print(classIds, bbox)
    for classIds, cofidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
        cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
        cv2.putText(img, classNames[classIds - 1], (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1,
                    (0, 255, 0), 2)
    cv2.imshow("Output", img)
    cv2.waitKey(0)



def image():
    img = cv2.imread('traffic-1024x800.jpg')
    classNames = []
    classFile = 'coco.names'
    with open(classFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')

    configpath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weigthpath = 'frozen_inference_graph.pb'

    net = cv2.dnn_DetectionModel(weigthpath, configpath)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)
    classIds, confs, bbox = net.detect(img, confThreshold=0.55)
    print(classIds, bbox)
    for classIds, cofidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
        cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
        cv2.putText(img, classNames[classIds - 1], (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1,
                    (0, 255, 0), 2)
    cv2.imshow("Output", img)
    cv2.waitKey(0)



def write_slogan():
    print("Tkinter is easy to use!")

    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap.set(3, 640)
    cap.set(4, 480)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open video")
    classNames = []
    classFile = 'coco.names'
    with open(classFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')

    configpath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weigthpath = 'frozen_inference_graph.pb'

    net = cv2.dnn_DetectionModel(weigthpath, configpath)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)
    while True:
        ret, img = cap.read()
        classIds, confs, bbox = net.detect(img, confThreshold=0.55)
        print(classIds, bbox)
        if len(classIds) != 0:

            for classIds, cofidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                cv2.putText(img, classNames[classIds - 1], (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1,
                            (0, 255, 0), 2)
            cv2.imshow("Output", img)
            cv2.waitKey(1)

root = tk.Tk()

root.geometry("200x100")
frame = tk.Frame(root)
frame.pack()
button = tk.Button(frame,text="QUIT",fg="red",command=quit)
button.pack()
slogan = tk.Button(frame,text="Start With camra",command=write_slogan)
slogan.pack()
'''button = tk.Button(frame,text="Object Detection form Image",command=image)
button.pack(side=tk.LEFT)'''

button = Button(root, text = "Load Image", command = clicked)
button.pack()
root.mainloop()