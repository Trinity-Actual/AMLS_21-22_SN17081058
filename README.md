# AMLS_21-22-_SN17081058
This repository contains code which is used to solve the 2021/22 AMLS coursework task.
The repository was remade due to pull request error in original repository. Therefore initial commit for this repository contains initial partially started code.

# 1. Overview of Task
The task given is an image classification task regarding Magnetic Resonance Imaging (MRI) images of various brain scans. This is based off the problem on Kaggle regarding [Brain Tumor Classification](https://www.kaggle.com/sartajbhuvaji/brain-tumor-classification-mri). Given an initial dataset of 3000 MRI images with their respective labels containing four unique classes, no_tumor, glioma_tumor, meningioma_tumor and pituitary_tumor. We are to train, validate and test machine learning or deep learning models to solve two problems surrounding the given dataset as follows:

**1. Binary Classification**
    -Build a classifier to identify whether there is a tumor in the MRI image dataset.
**2. Multi-class Classification**
    -Build a classifier to identify the type of tumor in each image, meningioma tumor, glioma tumor, pituitary tumor or no tumor.

The task allows any model of any type (Supervised Learning, Unsupervised Learning or Deep Learning) to be used to solve either of the tasks, with the same model allowed to be used for both tasks. However, one non-deep learning model must be implemented.

The models created in this repository are Support Vector Machine (SVM) and Convolutional Neural Network (CNN) based on the AlexNet architecture. Both models were trained for the binary and multi-class classification task.

# 2. Requirements
The `requirements.txt` file present in the repository lists all Python libraries used in the various notebooks and are required to run properly. The necessary requirements can be installed by opening a terminal, changing directory to the workspace containing the repository after cloning and using:
```
pip install -r requirements.txt
```
From the requirements file the libraries used are:
-Keras == 2.3.1
-ipynb == 0.5.1
-matplotlib == 3.4.3
-numpy == 1.21.2
-opencv_python_headless == 4.5.4.60
-pandas == 1.3.4
-scikit_learn == 1.0.1
-tensorflow == 2.1.0
-tqdm == 4.62.3
# 3. Organisation of Repository


# 4. Role of each File
### 4.1 functions 
### 4.2 Data-Preprocessing
### 4.3 Binary-Task-SVM
### 4.4 Binary-Task-CNN
### 4.5 Multiclass-Task-SVM
### 4.6 Multiclass-Task-CNN
### 4.7 Dataset Files
### 4.8 Model files
### 4.9 Model_Evaulation Files


# 5. Running Code to obtain results
