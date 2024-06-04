import cv2 as cv
import numpy as np
import glob
import os
import config

def get_img_paths(folder_name):
    '''
    return list of paths found in a folder (non-recursive)
    Args:
        - folder_name: name of the folder inside dataset path
    Return:
        - fractured_files: list of fractured image file paths
        - non_fractured_files: list of non fractured image file paths
    '''
    folder_path = config.DATASET_PATH + folder_name + '/'
    fractured_files = glob.glob(os.path.join(folder_path + 'fractured', '*'))
    nfractured_files = glob.glob(os.path.join(folder_path + 'not fractured', '*'))
    return fractured_files, nfractured_files

def prep_data(files, label, debug = False):
    '''
    read a list of files with designated label and return np array of img files and corresponding label
    Args:
        - files: a list of image file paths
        - label: 0 vs 1 (not fractured vs fractured)
    Return:
        - data: numpy array of valid image files
        - label: numpy array of corresponding label

    Sample code:
        x_1, y_1 = prep_data(fractured_files, 1)
        x_0, y_0 = prep_data(nfractured_files, 0)
        x_train = np.concatenate([x_0,x_1], axis=0)
        y_train = np.concatenate([y_0,y_1], axis=0)
        x_train.shape, y_train.shape
    '''
    data = []
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            img = cv.imread(file)
            if img is not None:
                height, width = img.shape[:2]
                if height == 224 and width == 224:
                    prep = cv.cvtColor(img, cv.COLOR_BGR2GRAY).flatten()
                    data += [prep]
                else:
                    if debug:
                        print(f"Discarding {file}: size is {width}x{height}")
    return np.array(data), np.array([label]*len(data)).reshape(-1,1)