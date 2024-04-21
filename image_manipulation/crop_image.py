import sys

import cv2
import numpy as np

if len(sys.argv) < 2:
    print("Usage: python crop_image.py <filename>")
    exit(0)

img = cv2.imread(sys.argv[1])
cropped_img = img[0:800, 0:800]
cv2.imshow("Cropped Image", cropped_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
