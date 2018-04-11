"""
Copyright (c) <2018> YoongiKim

 See the file license.txt for copying permission.
"""

import cv2
import numpy as np
import os
from RiderEnvironment import tf_mnist_custom

last_rect_sum = 0
file_counter = 0

def read(img):
    global last_rect_sum

    lower = np.array([250,250,250])
    upper = np.array([255,255,255])
    white = cv2.inRange(img, lower, upper)

    image, contours, hierachy = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])
    #image = cv2.drawContours(img, contours, -1, (0,0,255), 1)
    #print(len(contours))

    # if(len(contours)==0 or sum(cv2.boundingRect(contours[-1])) == last_rect_sum):
    #    return

    digits = []
    visualize = np.zeros((100, 100, 3), np.uint8)

    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        #print("x{} y{} w{} h{}".format(x, y, w, h))
        if(23<=h<=29 and w<=24):
            crop = white[y:y + h, x:x + w]
            finalImg = process_image(crop)
            digits.append(tf_mnist_custom.mnist_predict(finalImg))

            visualize = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)

    score_str = ""
    for i in range(len(digits)):
        score_str = score_str + str(digits[i])
    try:
        score = int(score_str)
    except ValueError:
        score = -10

    visualize = put_text(str(score), visualize)
    show(visualize, "score", 313, 0)

    # last_rect_sum = sum(cv2.boundingRect(contours[-1]))

    return score


def process_image(img):
    #print("{}, {}".format(np.size(img, 1), np.size(img, 0)))
    w = np.size(img, 1)
    h = np.size(img, 0)

    img = cv2.resize(img, None, fx=20/h, fy=20/h, interpolation=cv2.INTER_NEAREST)

    w = int(w*20/h)
    h = int(h*20/h)
    x_padding = int((28 - w)/2)
    y_padding = int((28 - h) / 2)

    img = cv2.copyMakeBorder(img, y_padding, y_padding, x_padding, x_padding, cv2.BORDER_CONSTANT, None, 0)
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_NEAREST)
    #show(img, "digit", 300, 100)

    #global  file_counter
    #write_img(img, os.getcwd() +"/score_train_data/"+ str(file_counter) + ".png")
    #file_counter += 1

    img = to_binary(img)
    return img

def to_binary(img):
    retval, threshold = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
    return np.array(threshold)

def put_text(text, img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (5,32)
    fontScale = 1
    fontColor = (0, 0, 255)
    lineType = 2

    return cv2.putText(img, text,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)

def show(img, title="score", x=400, y=500):
    cv2.imshow(title, img)
    cv2.moveWindow(title, x, y)
    cv2.waitKey(1)

def write_img(img, file_name):
    cv2.imwrite(file_name, img)