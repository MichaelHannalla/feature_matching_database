import cv2
import numpy as np
from utils import check_color_feaasibility

img1 = cv2.imread('sample1/a01166.jpg')
img2 = cv2.imread('sample2/a01186.jpg')

orb = cv2.ORB_create(nfeatures= 1000)

kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

good_matches = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good_matches.append([m])

print("Number of good feature matches: {}".format(len(good_matches)))
print("Is the match feasible: {}".format(check_color_feaasibility(img1, img2)))
matches_img = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None, flags=2)

matches_img = cv2.resize(matches_img, (matches_img.shape[1]//4, matches_img.shape[0]//4))
cv2.imshow('matches', matches_img)
cv2.waitKey(0)