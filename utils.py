import os
import scipy
import scipy.misc
import scipy.cluster
import cv2
import numpy as np
import xlsxwriter
from PIL import Image
from tqdm import tqdm
from itertools import compress

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
    
def read_image(path):
    img = Image.open(path)
    img = np.array(img)                                       # convert from PIL image to openCV
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def load_descriptors(imgs_folder, extractor):
    
    print("Loading descriptors from {} folder.".format(imgs_folder))
    list = os.listdir(imgs_folder)      # Get everything in the folder directory
    des_names_list = []                 # List that will hold the descriptor and the corresponding file name

    for idx, img_path in enumerate(tqdm(list)):
        
        # Read every image in samples folder and database folder, do same conversions on sample image
        img = read_image(imgs_folder + "/" + img_path)
        kp, des = extractor.detectAndCompute(img, None)
        img_resized = cv2.resize(img, (img.shape[1]//3, img.shape[0]//3))
        des_names_list.append((des, img_path, img_resized))
    
    return des_names_list

def write_results_excel(filename, datalist, use_bruteforce= True):
    
    if use_bruteforce:
        excel_suffix = "bf"
    else:
        excel_suffix = "flann"

    with xlsxwriter.Workbook(filename.format(excel_suffix)) as workbook:              # excel writing session
        worksheet = workbook.add_worksheet()

        for row_num, data in enumerate(datalist):
            worksheet.write_row(row_num, 0, data)

def adjust_name(unadjusted_name):
    ret = unadjusted_name.split('.')
    label = ret[0]      # removed the extension
    fragments = label.split("_")
    adjusted_name = fragments[0]
    return adjusted_name

def is_same_ref_name(name1, name2):
    name1_ref = name1.split("-")[0]
    name2_ref = name2.split("-")[0]
    if name1_ref == name2_ref:
        return True
    else:
        return False
    