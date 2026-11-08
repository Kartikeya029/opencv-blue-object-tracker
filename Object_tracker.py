import cv2
import numpy as np

camera = cv2.VideoCapture(0)

lower_blue = np.array([90, 50, 30], dtype = np.uint8)
upper_blue = np.array([140, 255, 255], dtype = np.uint8)
    
while True:
    success, frame = camera.read()
    if not success:
        break
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    kernel = np.ones((5,5), np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, hierarchy = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    areas=[]
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 0:
            areas.append(area)
    output = cv2.cvtColor(opening, cv2.COLOR_GRAY2BGR)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        print(cv2.contourArea(largest))
        cv2.drawContours(output, [largest], -1, (0, 255, 0), 2)

        a, b, c, d = cv2.boundingRect(largest)
        cv2.rectangle(output, (a, b), (a+c, b+d), (0, 0, 255), 2)

        M = cv2.moments(largest)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            print("Centre X:", cx)
            print("Centre Y:", cy)

            circle = cv2.circle(output, (cx, cy), 5, (255, 0, 0), -1)

    cv2.imshow("Blue Detection", output)
    cv2.imshow("Object Tracker", frame)

    if cv2.waitKey(1) == ord('p'):
        break
print(mask.shape, mask.dtype)


camera.release()
cv2.destroyAllWindows()









