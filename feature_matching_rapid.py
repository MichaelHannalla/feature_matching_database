import cv2
import numpy as np
import os
from numpy.lib.npyio import load
import xlsxwriter
from PIL import Image
from utils import check_color_feaasibility, load_descriptors, read_image
from tqdm import tqdm

def main():
    #TODO: using args
    
    good_matches_threshold = 60         # Threshold for matches between images to be considered same item
    use_bruteforce = True               # Choosing wheter to use the brute force matcher or flann matcher
    one_by_one = True                   # Choosing whether we'll compare each two combinations of images and get the best not just stop by the first good match

    samples_folder = "sample1"          # Path for the samples folder
    database_folder = "sample2"         # Path for the database folder
    
    correspondences = [['sample', 'database', 'match score']]                # Empty list to hold correspondences, will later be converted to an excel file

    samples_list = os.listdir(samples_folder)       # Get everything in the samples folder directory
    database_list = os.listdir(database_folder)     # Get everything in the database folder directory

    # Create the feature extractors
    orb = cv2.ORB_create(nfeatures= 1000)           # Create the orb feature extractor

    # Create the matchers
    bf = cv2.BFMatcher()                                                        # create the brute-force matcher
    index_params = dict(algorithm = 0, trees = 5)
    search_params = dict()                                                      # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)                  # create the flann matcher

    database_data = load_descriptors(database_folder, orb)
    print("Finished loading all the database descriptors")
    
    for sample_idx, sample_img_path in enumerate(tqdm(samples_list)):

        curr_max_match_number = good_matches_threshold                              # Initialize a variable that holds the max matches of all

        print("_Comparing sample {}/{}.".format(sample_idx, len(samples_list)))
        # Open the sample image, then convert to openCV type, then change color space.
        sample_img = read_image(samples_folder + "/" + sample_img_path)
        sample_kp, sample_des = orb.detectAndCompute(sample_img, None)

        for database_point in database_data:
            database_des, database_name, database_img = database_point

            if use_bruteforce:
                matches = bf.knnMatch(sample_des, database_des, k=2)                          # perform knn matching with 2 nearest neighbours
            else:
                pass
                print("Feature unsupported at current version release")
                raise NotImplementedError
                #des1 = des1.astype(np.float32)
                #des2 = des2.astype(np.float32)
                #matches = flann.knnMatch(des1, des2, k=2)

            good_matches = []                                                   # list of matches
            for m, n in matches:
                if m.distance < 0.75*n.distance:
                    good_matches.append([m])    

            if len(good_matches) > good_matches_threshold:

                if not check_color_feaasibility(sample_img, database_img):
                    continue

                if one_by_one:
                    if len(good_matches) > curr_max_match_number:
                        curr_max_match_number = len(good_matches)
                        curr_max_match = database_name

                if not one_by_one:
                    print("Match found between {} in samples folder and {} in database folder".format(sample_img_path, database_name))
                    correspondences.append([sample_img_path, database_name, len(good_matches)])
                    break                                                       # break and stop searching for this if a match is found (this to allow faster search)
            
        if (curr_max_match_number > good_matches_threshold) and one_by_one:
            print("Match found between {} in samples folder and {} in database folder".format(sample_img_path, curr_max_match))
            correspondences.append([sample_img_path, curr_max_match, curr_max_match_number])

        else:    
            print("No match found for sample {} in the database folder".format(sample_img_path))
            correspondences.append([sample_img_path, "NO MATCH", 0])

    if use_bruteforce:
        excel_suffix = "bf"
    else:
        excel_suffix = "flann"

    with xlsxwriter.Workbook('excels/correspondences_{}_rapid.xlsx'.format(excel_suffix)) as workbook:         # excel writing session
        worksheet = workbook.add_worksheet()

        for row_num, data in enumerate(correspondences):
            worksheet.write_row(row_num, 0, data)

if __name__ == "__main__":
    main()