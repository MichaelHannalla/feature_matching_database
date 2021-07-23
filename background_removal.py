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
    low = 180
    high = 250
    min_area = 0
    max_area = 1
    mask_color = (0, 0, 0)
    mask_dilate_iter = 10
    mask_erode_iter = 10
    blur = 5

    img = cv2.imread('sample_data/a01000.jpg')
    img = cv2.resize(img, (img.shape[1]//3, img.shape[0]//3))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(img_gray, low, high, L2gradient=True)
    edges = cv2.dilate(edges, None)
    edges = cv2.erode(edges, None)

    # get the contours and their areas
    contour_info = [(c, cv2.contourArea(c),) for c in cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]]

    image_area = img.shape[0] * img.shape[1]
    # calculate max and min areas in terms of pixels
    max_area = max_area * image_area
    min_area = min_area * image_area

    # Set up mask with a matrix of 0's
    mask = np.zeros(edges.shape, dtype = np.uint8)

    for contour in contour_info:
        # Instead of worrying about all the smaller contours, if the area is smaller than the min, the loop will break
        if contour[1] > min_area and contour[1] < max_area:
            # Add contour to mask
            mask = cv2.fillConvexPoly(mask, contour[0], (255))
            
    # use dilate, erode, and blur to smooth out the mask
    mask = cv2.dilate(mask, None, iterations= mask_dilate_iter)
    mask = cv2.erode(mask, None, iterations= mask_erode_iter)
    mask = cv2.GaussianBlur(mask, (blur, blur), 0)
    mask_stack = np.stack((mask, mask, mask), axis=2)
    masked = cv2.bitwise_and(img, mask_stack)
    masked = cv2.erode(masked, kernel=(25,25,25))
    cv2.imshow('frame', masked)
    cv2.waitKey(0)

if __name__=="__main__":
    main()