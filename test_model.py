
'''
Created on 2019年3月30日

@author: 余创
'''
import os
import sys
import random
from PIL import Image
import math
import numpy as np
import skimage.io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import time
from mrcnn.config import Config
from datetime import datetime 
# Root directory of the project
start=time.clock()

ROOT_DIR = os.getcwd()

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
from samples.coco import coco


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(MODEL_DIR ,"mask_rcnn_shapes_0080.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
    print("cuiwei***********************")

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
IMAGE_DIR2 = os.path.join(ROOT_DIR, "test_results4")


def divide_method1(img,w,h,m,n):#分割成m行n列
    gx, gy = np.meshgrid(np.linspace(0, w, n),np.linspace(0, h, m))
    gx=np.round(gx).astype(np.int)
    gy=np.round(gy).astype(np.int)

    divide_image = np.zeros([m-1, n-1, int(h*1.0/(m-1)+0.5), int(w*1.0/(n-1)+0.5),3], np.uint8)#这是一个五维的张量，前面两维表示分块后图像的位置（第m行，第n列），后面三维表示每个分块后的图像信息
    for i in range(m-1):
        for j in range(n-1):      
            divide_image[i,j,0:gy[i+1][j+1]-gy[i][j], 0:gx[i+1][j+1]-gx[i][j],:]= img[
                gy[i][j]:gy[i+1][j+1], gx[i][j]:gx[i+1][j+1],:]#这样写比a[i,j,...]=要麻烦，但是可以避免网格分块的时候，有些图像块的比其他图像块大一点或者小一点的情况引起程序出错
            im = Image.fromarray(divide_image[i,j,:])
            new_name=str((i*(n-1)+j)).zfill(3)+'.jpg'
            new_mask = os.path.join(IMAGE_DIR2,new_name)
            im.save(new_mask)
 
    return divide_image


'''
def display_blocks(divide_image):#    
    m,n=divide_image.shape[0],divide_image.shape[1]
    for i in range(m):
        for j in range(n):
            
            plt.subplot(m,n,i*n+j+1)
            plt.imshow(divide_image[i,j,:])
            plt.axis('off')
            plt.title('block:'+str(i*n+j+1))
    
 '''   

def image_compose(IMAGE_ROW,IMAGE_COLUMN,IMAGE_length,IMAGE_width,IMAGES_PATH,image_names):
    to_image = Image.new('RGB', (IMAGE_COLUMN * IMAGE_length, IMAGE_ROW * IMAGE_width)) #创建一个新图
    # 循环遍历，把每张图片按顺序粘贴到对应位置上
    for y in range(1, IMAGE_ROW + 1):
        for x in range(1, IMAGE_COLUMN + 1):
            from_image = Image.open(IMAGES_PATH + image_names[IMAGE_COLUMN * (y - 1) + x - 1]).resize(
                (IMAGE_length, IMAGE_width),Image.ANTIALIAS)
            to_image.paste(from_image, ((x - 1) * IMAGE_length, (y - 1) * IMAGE_width))
    return to_image.save('test_results6/final.jpg') # 保存新图



class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 2  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 1600

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8 * 6, 16 * 6, 32 * 6, 64 * 6, 128 * 6)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE =100

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50

#import train_tongue
#class InferenceConfig(coco.CocoConfig):
class InferenceConfig(ShapesConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()

model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)


# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG','box_pane', 'circle']
# Load a random image from the images folder
'''
file_names = next(os.walk(IMAGE_DIR))[2]
image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
'''

img = cv2.imread(r'20666.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
print( '\t\t\t  原始图像形状:\n', '\t\t\t',img.shape ) 
h, w = img.shape[0], img.shape[1]

#原始图像分块
m=6
n=6
divide_image1=divide_method1(img,w,h,m+1,n+1)#该函数中m+1和n+1表示网格点个数，m和n分别表示分块的块数
print(divide_image1.shape)
#fig2 = plt.figure('分块后的子图像:四舍五入法')
#display_blocks(divide_image1)
  
sum_pixels=0
count = os.listdir(IMAGE_DIR2)
for i in range(0,len(count)):
    path = os.path.join(IMAGE_DIR2, count[i])
    if os.path.isfile(path):
        file_names = next(os.walk(IMAGE_DIR2))[2]
        image = skimage.io.imread(os.path.join(IMAGE_DIR2, count[i]))
        # Run detection
        results = model.detect([image], verbose=1)
        r = results[0]
        return_pixels=visualize.display_instances(count[i],image, r['rois'], r['masks'], r['class_ids'], 
                            class_names, r['scores'])
    sum_pixels=	sum_pixels+ return_pixels
	
print("The totle pixel points are {}" .format(sum_pixels))							
#分块图像还原
IMAGES_PATH = 'test_results6/'
IMAGE_length=1739
IMAGE_width=1760
image_names = [name for name in os.listdir(IMAGES_PATH)]
if len(image_names) != m * n:
    raise ValueError("合成图片的参数和要求的数量不能匹配！")
image_compose(m,n,IMAGE_length,IMAGE_width,IMAGES_PATH,image_names)

elapsed=(time.clock()-start)
print("Time used:",elapsed)