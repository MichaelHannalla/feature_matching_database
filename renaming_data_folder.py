# File: renaming_data_folder.py
# Author: @MichaelHannalla
# Project: OCR on a database and template matching
# Description: Python file for preparing and renaming the samples for ease of making a database

import os
from numpy.lib.index_tricks import ix_
import xlsxwriter
import numpy as np
import pandas as pd
from PIL import Image

def main():

    #TODO: using args
    data_path = "sample_data"
    renamed_path = "sample_data_renamed"
    excel_names_file = 'names.xlsx'
    
    names_df = pd.read_excel(excel_names_file, engine='openpyxl')       # pandas dataframe to import excel file
    names = names_df['NAMES'].to_list()                                 # converting to a python list
    
    data = os.listdir(data_path)                                        # list images in the data path
    changelog = []                                                      # changelog list
    dup_idx = 0

    for idx, img_path in enumerate(data):
        
        img = Image.open(data_path + "/" + img_path)                # open the sample image
        curr_new_name = str(names[idx] + ".jpg")                    # get its target new name

        # To handle duplicate names
        try:
            Image.open(renamed_path + "/" + curr_new_name)          # try to open if some file exists with the same name, will raise FileNotFoundError if not found
            dup_idx += 1 
            modified_curr_new_name = str(names[idx] + "-{}.jpg".format(dup_idx))    # modify the new name to hold duplicates
            img.save(renamed_path + "/" + modified_curr_new_name)                   # save with the modified name
            changelog.append([img_path, modified_curr_new_name])                    # append to the changelog
            
        except FileNotFoundError:
            # this is a new name
            #dup_idx = 0
            img.save(renamed_path + "/" + curr_new_name)            # save with the target new name because it's unique
            changelog.append([img_path, curr_new_name])             # append to the changelog


    with xlsxwriter.Workbook('changelog.xlsx') as workbook:         # excel writing session
        worksheet = workbook.add_worksheet()

        for row_num, data in enumerate(changelog):
            worksheet.write_row(row_num, 0, data)

if __name__ == "__main__":
    main()