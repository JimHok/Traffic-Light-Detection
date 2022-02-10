import cv2
import numpy as np

cap = cv2.VideoCapture(r"C:\Users\jimyj\Desktop\GitHub\Traffic-Light-Detection\Test Video\Test7.mp4")

while True:
    _, frame = cap.read()
    original = frame.copy()
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Red color
    low_red0 = np.array([0, 100, 100])
    high_red0 = np.array([10, 255, 255])
    low_red = np.array([160, 100, 100])
    high_red = np.array([179, 255, 255])
    red_mask = cv2.GaussianBlur(cv2.inRange(hsv_frame, low_red, high_red), (3, 3), 1)

    red = cv2.bilateralFilter(cv2.bitwise_and(frame, frame, mask=red_mask), 11, 75, 75)

    cnts = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]


    for c in cnts:

        # Circle Detection
        img = cv2.cvtColor(red, cv2.COLOR_BGR2GRAY)
        rcircle = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
                                    param1=50,param2=10,minRadius=3,maxRadius=10)

        if rcircle is not None:
            rcircle = np.uint16(np.around(rcircle))
            for i in rcircle[0,:]:
                # draw the outer circle
                cv2.circle(original,(i[0],i[1]),i[2]*2,(40,40,255),3)
                # draw the outer circle
                cv2.circle(red,(i[0],i[1]),i[2]*2,(40,40,255),3)



    # Yellow color
    low_yell = np.array([20,100,100])
    high_yell = np.array([40,255,255])
    yell_mask = cv2.medianBlur(cv2.inRange(hsv_frame, low_yell, high_yell), 3)
    yell = cv2.medianBlur(cv2.bitwise_and(frame, frame, mask=yell_mask), 3)

    cnts = cv2.findContours(yell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    for c in cnts:

        # Circle Detection
        img = cv2.cvtColor(yell, cv2.COLOR_BGR2GRAY)
        ycircle = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
                                    param1=50,param2=15,minRadius=3,maxRadius=10)

        if ycircle is not None:
            ycircle = np.uint16(np.around(ycircle))
            for i in ycircle[0,:]:
                # draw the outer circle
                cv2.circle(original,(i[0],i[1]),i[2]*2,(40,255,255),2)
                # draw the outer circle
                cv2.circle(yell,(i[0],i[1]),i[2]*2,(40,255,255),3)


    # Green color
    low_green = np.array([40, 100, 100])
    high_green = np.array([95, 255, 255])
    green_mask = cv2.GaussianBlur(cv2.inRange(hsv_frame, low_green, high_green), (7, 7), 0)
    green = cv2.bilateralFilter(cv2.bitwise_and(frame, frame, mask=green_mask), 7, 75, 75)

    cnts = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    for c in cnts:

        # Circle Detection
        img = cv2.cvtColor(green, cv2.COLOR_BGR2GRAY)
        gcircle = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
                                    param1=50,param2=25,minRadius=3,maxRadius=13)

        if gcircle is not None:
            gcircle = np.uint16(np.around(gcircle))
            for i in gcircle[0,:]:
                # draw the outer circle
                cv2.circle(original,(i[0],i[1]),i[2]*2,(40,255,40),2)
                # draw the outer circle
                cv2.circle(green,(i[0],i[1]),i[2]*2,(40,255,40),3)



    final = cv2.vconcat([cv2.hconcat([original, green]),cv2.hconcat([yell, red])])

    cv2.imshow("I", final)


    key = cv2.waitKey(1)
    if key == 27:
        break