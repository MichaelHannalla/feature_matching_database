from os import remove
import scipy
import scipy.misc
import scipy.cluster
import cv2
import numpy as np
from PIL import Image


NUM_CLUSTERS = 5
COLOR_EUCLID_THRESH = 20000

def remove_background(img):
    # Parameters
    low = 25
    high = 25
    mask_color = (0, 0, 0)
    mask_dilate_iter = 10
    mask_erode_iter = 10
    blur = 5

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

    return out

def get_dominant_color(image):
    im = Image.fromarray(image)
    im = im.resize((150, 150))      # optional, to reduce time
    ar = np.asarray(im)
    shape = ar.shape
    ar = ar.reshape(scipy.product(shape[:2]), shape[2]).astype(float)

    codes, dist = scipy.cluster.vq.kmeans(ar, NUM_CLUSTERS)

    vecs, dist = scipy.cluster.vq.vq(ar, codes)         # assign codes
    counts, bins = scipy.histogram(vecs, len(codes))    # count occurrences

    index_max = scipy.argmax(counts)                    # find most frequent
    peak = codes[index_max]
    return peak

def check_color_feaasibility(query_img, train_img):
    query_img_noback = remove_background(query_img)
    train_img_noback = remove_background(train_img)
    query_img_dominant_color = get_dominant_color(query_img_noback)
    train_img_dominant_color = get_dominant_color(train_img_noback)
    euclid = np.linalg.norm(query_img_dominant_color**2 - train_img_dominant_color**2)
    if euclid > COLOR_EUCLID_THRESH:
        return euclid, False
    else:
        return euclid, True
    

    