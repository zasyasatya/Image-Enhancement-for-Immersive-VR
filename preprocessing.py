import cv2
from natsort import natsorted
import glob
import os

# os.makedirs("preprocessed_image")

for filename in natsorted(glob.glob('vr\\*.png'), key = len):

    pure_filename = filename.split("\\")[-1].split(".")[0]
    print(pure_filename)
    # print(filename)
    image = cv2.imread(filename)

    y=0
    x=72
    h=1920
    w=879
    crop_image = image[x:w, y:h]
    resize_image = cv2.resize(crop_image, (1280, 720))
    lab_image = cv2.cvtColor(resize_image, cv2.COLOR_BGR2LAB)
    cv2.imwrite(f"preprocessed_image_lab\\{pure_filename}.png", lab_image)
