# feature_matching_database
This repo is for object classification of samples against a predefined database using the ORB feature extractor. The main code checks each sample against the whole database and if a correspondence is found it's added to an excel file. This excel file is written at the end of the code for further use.

## Setting up and running
1. Clone this repo. \
2. Install python requirements by `pip install -r requirements.txt`. <br/>
3. Create two directories, the first one is for samples and the other for database. <br/>
4. Go to the `feature_matching.py` code and adjust your paths accordingly. <br/>
5. Run the code and get the matches in `excels/correspondences.xlsx` file. <br/>
