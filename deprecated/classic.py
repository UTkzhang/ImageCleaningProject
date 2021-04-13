import cv2
import os
import numpy as np

for root, dirs, files in os.walk("./data", topdown=False):
    for name in files:
        print(os.path.join(root, name))
    for name in dirs:
        print(os.path.join(root, name))

data_path = "./datasets/noisytext/testA"
results_path = "./results"

########################################################
# PART 1 - Median Filtering
########################################################
def median_filter(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # Load image
    img_median = cv2.medianBlur(img, 25) # Add median filter to image
    result = np.minimum(img.astype(np.uint16)+(255-img_median.astype(np.uint16)), 255)
    return result.astype(np.uint8)


########################################################
# PART 2 - Edge Detection, Dilation Erosion
########################################################
def edge_dilation_erosion_filter(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # Load image
    edges = cv2.Canny(img,100,200)
    dilated = cv2.dilate(edges, np.ones((3,3),np.uint8), iterations=1)
    eroded = cv2.erode(dilated, np.ones((4,4),np.uint8), iterations=1)
    return cv2.bitwise_not(eroded)


########################################################
# PART 3 - Adaptive Filter
########################################################
def adaptive_filter(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # Load image
    return cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,15,20)


if not os.path.exists(results_path+"/median_filtered"):
    os.mkdir(results_path+"/median_filtered")
if not os.path.exists(results_path+"/edge_dilation_erosion"):
    os.mkdir(results_path+"/edge_dilation_erosion")
if not os.path.exists(results_path+"/adaptive_filtered"):
    os.mkdir(results_path+"/adaptive_filtered")

for img in sorted(os.listdir(data_path)):
    if img.endswith('.png'):
        print(os.path.join(data_path,img))
        cv2.imwrite(os.path.join(results_path+"/median_filtered",img),median_filter(os.path.join(data_path,img)))
        cv2.imwrite(os.path.join(results_path+"/edge_dilation_erosion",img),edge_dilation_erosion_filter(os.path.join(data_path,img)))
        cv2.imwrite(os.path.join(results_path+"/adaptive_filtered",img),adaptive_filter(os.path.join(data_path,img)))
