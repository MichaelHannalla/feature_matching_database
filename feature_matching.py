import cv2
import numpy as np
import os
from PIL import Image

good_matches_threshold = 10

folder1_path = "/home/heidi/Feature Matching Project/sample"
folder2_path = "/home/heidi/Feature Matching Project/sample2"

#img1 = cv2.imread('sample/a01000.jpg')
#img2 = cv2.imread('sample2/a01010.jpg')


img_1 = os.listdir(folder1_path)
img_2 = os.listdir(folder2_path)

#loop on every image in both folders
for img_i_path in img_1:
    for img_j_path in img_2:
        #read everyimage in folder 1 and folder 2
        img1 = Image.open(folder1_path + "/" + img_i_path)
        img2 = Image.open(folder1_path + "/" + img_j_path)

        # convert from PIL image to openCV 
        open_cv_image_1 = np.array(img1) 
        open_cv_image_2 = np.array(img2) 
        img1 = cv2.cvtColor(open_cv_image_1, cv2.COLOR_BGR2RGB) 
        img2 = cv2.cvtColor(open_cv_image_2, cv2.COLOR_BGR2RGB) 

        orb = cv2.ORB_create(nfeatures= 1000)

        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        good_matches = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good_matches.append([m])

                if len(good_matches) > good_matches_threshold:
                    print(img_i_path, img_j_path)

        matches_img = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None, flags=2)


        matches_img = cv2.resize(matches_img, (matches_img.shape[1]//5, matches_img.shape[0]//5))
        #cv2.imshow('matches', matches_img)
        #cv2.waitKey(0)