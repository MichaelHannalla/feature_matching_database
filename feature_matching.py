import cv2
import numpy as np
import os
import xlsxwriter
from PIL import Image

def main():
    #TODO: using args
    
    good_matches_threshold = 40         # Threshold for matches between images to be considered same item
    use_bruteforce = False

    samples_folder = "sample1"          # Path for the samples folder
    database_folder = "sample2"         # Path for the database folder

    correspondences = []                # Empty list to hold correspondences

    samples_list = os.listdir(samples_folder)       # Get everything in the samples folder directory
    database_list = os.listdir(database_folder)     # Get everything in the database folder directory

    # Create the feature extractors
    orb = cv2.ORB_create(nfeatures= 1000)           # Create the orb feature extractor

    # Create the matchers
    bf = cv2.BFMatcher()                                                        # create the brute-force matcher
    index_params = dict(algorithm = 0, trees = 5)
    search_params = dict()                                                      # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)                  # create the flann matcher

    # Loop on every combination of images in both folders
    for idx1, sample_img_path in enumerate(samples_list):
        
        # Open the sample image, then convert to openCV type, then change color space.
        sample_img = Image.open(samples_folder + "/" + sample_img_path)
        sample_img = np.array(sample_img)   # convert from PIL image to openCV
        sample_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB) 

        # Loop on each image in the database
        for idx2, database_img_path in enumerate(database_list):

            # Read every image in samples folder and database folder, do same conversions on sample image
            database_img = Image.open(database_folder + "/" + database_img_path)
            database_img = np.array(database_img)    # convert from PIL image to openCV
            database_img = cv2.cvtColor(database_img, cv2.COLOR_BGR2RGB) 

            kp1, des1 = orb.detectAndCompute(sample_img, None)              # detect keypoints and their descriptors
            kp2, des2 = orb.detectAndCompute(database_img, None)

            if use_bruteforce:
                matches = bf.knnMatch(des1, des2, k=2)                      # perform knn matching with 2 nearest neighbours
            else:
                des1 = des1.astype(np.float32)
                des2 = des2.astype(np.float32)
                matches = flann.knnMatch(des1, des2, k=2)

            good_matches = []                                           # list of matches
            for m,n in matches:
                if m.distance < 0.75*n.distance:
                    good_matches.append([m])    

            if len(good_matches) > good_matches_threshold:
                print("Match found between {} in samples folder and {} in database folder".format(sample_img_path, database_img_path))
                correspondences.append([sample_img_path, database_img_path])
                break                                                   # break and stop searching for this if a match is found (this to allow faster search)

    if use_bruteforce:
        excel_suffix = "bf"
    else:
        excel_suffix = "flann"

    with xlsxwriter.Workbook('excels/correspondences_{}.xlsx'.format(excel_suffix)) as workbook:         # excel writing session
        worksheet = workbook.add_worksheet()

        for row_num, data in enumerate(correspondences):
            worksheet.write_row(row_num, 0, data)

if __name__ == "__main__":
    main()