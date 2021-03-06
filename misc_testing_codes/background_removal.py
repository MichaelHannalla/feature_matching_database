import cv2
import numpy as np

def nothing(x):
    pass

def main():
    #cv2.namedWindow('Trackbars')
    #cv2.createTrackbar("LOW", "Trackbars", 255, 255, nothing)
    #cv2.createTrackbar("HIGH", "Trackbars", 255, 255, nothing)

    # Parameters
    low = 25
    high = 25
    mask_color = (0, 0, 0)
    mask_dilate_iter = 10
    mask_erode_iter = 10
    blur = 5

    img = cv2.imread('sample1/a01046.jpg')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(img_gray, low, high, L2gradient=True)
    edges = cv2.dilate(edges, None)
    edges = cv2.erode(edges, None)

    # get the contours and their areas
    contours = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]

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
    masked = cv2.erode(masked, kernel=(101,101,101))

    # Now crop
    (y, x) = np.where(mask == 255)
    (topy, topx) = (np.min(y), np.min(x))
    (bottomy, bottomx) = (np.max(y), np.max(x))
    out = masked[topy:bottomy+1, topx:bottomx+1]

    out = cv2.resize(out, (out.shape[1]//3, out.shape[0]//3))
    cv2.imshow('masked', out)
    cv2.waitKey(0)


if __name__=="__main__":
    main()