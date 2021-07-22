# File: renaming_data_folder.py
# Author: @MichaelHannalla
# Project: OCR on a database and template matching
# Description: Python file for preparing and renaming the samples for ease of making a database

import os
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
    #print(len(names))
    
    data = os.listdir(data_path)
    changelog = []
    for idx, img_path in enumerate(data):
        img = Image.open(data_path + "/" + img_path)
        curr_new_name = str(names[idx] + ".jpg")
        img.save(renamed_path + "/" + curr_new_name)
        changelog.append([img_path, curr_new_name])

    with xlsxwriter.Workbook('changelog.xlsx') as workbook:
        worksheet = workbook.add_worksheet()

        for row_num, data in enumerate(changelog):
            worksheet.write_row(row_num, 0, data)

if __name__ == "__main__":
    main()