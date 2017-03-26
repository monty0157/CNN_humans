import os
from PIL import Image
import numpy as np


pos_img_list_train = os.listdir('./INRIAPerson/train_64x128_H96/pos')
pos_img_path_train = './INRIAPerson/train_64x128_H96/pos/'
neg_img_list_train = os.listdir('./INRIAPerson/train_64x128_H96/neg')
neg_img_path_train = './INRIAPerson/train_64x128_H96/neg/'

def image_import_train():
    placeholder = []
    data = []
    for file in pos_img_list_train:
        img = Image.open(pos_img_path_train + file)
        img = img.convert('RGB')
        img.load()
        img = img.resize((96,160))
        image = np.asarray( img, dtype="float32" )
        placeholder.append(image)
    for file in neg_img_list_train * 3:
        img = Image.open(neg_img_path_train + file)
        img = img.convert('RGB')
        img.load()
        img = img.resize((96,160))
        image = np.asarray( img, dtype="float32" )
        placeholder.append(image)
    #SHUFFLE DATA: First picture is positive, second picture is negative and so on. Make sure to have all the positive images in the new list.
    for i in range(int(len(pos_img_list_train))):
        data.append(placeholder[i])
        value = int(len(placeholder)-(i+1))
        data.append(placeholder[value])
        i += 1

    return data

pos_img_list_test = os.listdir('./INRIAPerson/test_64x128_H96/pos')
pos_img_path_test = './INRIAPerson/test_64x128_H96/pos/'
neg_img_list_test = os.listdir('./INRIAPerson/test_64x128_H96/neg')
neg_img_path_test = './INRIAPerson/test_64x128_H96/neg/'

def image_import_test_pos():
    data = []
    for file in pos_img_list_test:
        img = Image.open(pos_img_path_test + file)
        img = img.convert('RGB')
        img.load()
        img = img.resize((96,160))
        image = np.asarray( img, dtype="float32" )
        data.append(image)

    return data

def image_import_test_neg():
    data = []
    for file in neg_img_list_test:
        img = Image.open(neg_img_path_test + file)
        img = img.convert('RGB')
        img.load()
        img = img.resize((96,160))
        image = np.asarray( img, dtype="float32" )
        data.append(image)

    return data
