import cv2
import numpy as np

cap = cv2.VideoCapture(r"C:\Users\jimyj\Desktop\Python\Practices\Test3.mp4")

while True:
    _, circle = cap.read()

    img = cv2.cvtColor(circle, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
                                param1=50,param2=30,minRadius=10,maxRadius=40)

    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        # draw the outer circle
        cv2.circle(circle,(i[0],i[1]),i[2],(255,0,0),2)
        # draw the center of the circle
        cv2.circle(circle,(i[0],i[1]),2,(255,0,0),3)

    cv2.imshow('detected circles',circle)

    key = cv2.waitKey(1)
    if key == 27:
        break


