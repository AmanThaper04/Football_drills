import cv2
from PIL import Image

from util import get_limits


black = [255, 255,255]  
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()

    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lowerLimit, upperLimit = get_limits(color=black)

    mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)

    

    cv2.imshow('frame', mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()