# File: background_removal.py
# Author: @MichaelHannalla
# Project: OCR on a database and template matching
# Description: Python file for testing background removal on target objects

import cv2
import numpy as np

def nothing(x):
    pass

def main():
    #cv2.namedWindow('Trackbars')
    #cv2.createTrackbar("LOW", "Trackbars", 255, 255, nothing)
    #cv2.createTrackbar("HIGH", "Trackbars", 255, 255, nothing)

    # Parameters
    low = 200
    high = 300
    min_area = 0
    max_area = 1
    mask_color = (0, 0, 0)
    mask_dilate_iter = 10
    mask_erode_iter = 10
    blur = 5

    img = cv2.imread('sample1/a01028.jpg')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(img_gray, low, high, L2gradient=True)
    edges = cv2.dilate(edges, None)
    edges = cv2.erode(edges, None)

    # get the contours and their areas
    contours = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]

    image_area = img.shape[0] * img.shape[1]
    # calculate max and min areas in terms of pixels
    max_area = max_area * image_area
    min_area = min_area * image_area

    # Set up mask with a matrix of 0's
    mask = np.zeros(edges.shape, dtype = np.uint8)
    
    # Get the max area contour
    c = max(contours, key = cv2.contourArea)
    mask = cv2.fillConvexPoly(mask, c, (255))
            
    # use dilate, erode, and blur to smooth out the mask
    mask = cv2.dilate(mask, None, iterations= mask_dilate_iter)
    mask = cv2.erode(mask, None, iterations= mask_erode_iter)
    mask = cv2.GaussianBlur(mask, (blur, blur), 0)
    mask_stack = np.stack((mask, mask, mask), axis=2)
    masked = cv2.bitwise_and(img, mask_stack)
    masked = cv2.erode(masked, kernel=(25,25,25))
    cv2.imwrite('templates/pic2.jpg', masked)

if __name__=="__main__":
    main()