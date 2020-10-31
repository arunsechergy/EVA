# Assignment 15A
Create dataset that contains depth map, surface planes, and bounding boxes for all the classes in the YoloV3 dataset

#### Group members:
Arun Kumar RN (arun.rvbe@gmail.com)

**For this assignment, you MUST have,**
1. Create two datasets and share a link to GDrive (publicly available to anyone) in this readme file.
2. Explain how these two datasets were created
3. Add the notebook file to the repo, which was used to create these datasets

#### Google Drive link for the dataset
**a. For Depth maps**
	https://drive.google.com/drive/folders/19cEuSDPOAi3uwGBcVaVvMFMLT0ILxgCI?usp=sharing
	
**b. For surface planes**
https://drive.google.com/drive/folders/121tKDkz4KNhhdOqDu3Nj2oI8lbnVIA5L?usp=sharing

**Dataset explanation**

- The input dataset size has 3590 images, with focus on hardhat workers and safety vest
- Monocular depth estimation relies on large and diverse training sets. 
- Due to the challenges associated with acquiring dense ground-truth depth across different environments at scale, a number of datasets with distinct characteristics and biases was created. 

**Things to remember while running these models:**
1. We need to create the depth map and surface planes, by running the MiDaS Model and planeRCNN model on our SafetyHelmet dataset. 
2. Since, there were 3590 images, the dataset use to fill up the memory
2. I had to manually use the python garbage collector to make sure we free the memory after every batch
