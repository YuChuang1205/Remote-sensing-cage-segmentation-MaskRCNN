
paper:  Segmentation and density statistics of mariculture cages from remote sensing images using mask R-CNN.
[链接](https://www.researchgate.net/publication/351315048_Segmentation_and_Density_Statistics_of_Mariculture_Cages_from_Remote_Sensing_Images_Using_Mask_R-CNN)
"train_model.py" is the file that performs the training.
"test_model.py" is the file that executes the test.
"train_data" folder is a training sample set.
"test_result6" is the folder where the segmentation result graph is saved.
"logs" is the folder that generates the trained model.


Before training the model, you should add the pre-training weight parameter file "mask_rcnn_coco.h5", 
create a folder named "image", "test_result" and "logs" under this project.