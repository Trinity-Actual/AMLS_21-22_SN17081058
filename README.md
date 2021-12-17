# AMLS_21-22_SN17081058
This repository contains code which is used to solve the 2021/22 AMLS coursework task.
The repository was remade due to pull request error in original repository. Therefore initial commit for this repository contains initial partially started code.

# Contents
1. [Overview of Task](#1-overview-of-task) 
2. [Requirements](#2-requirements)
3. [Organisation of Repository](#3-organisation-of-repository)
4. [Role of Each File](#4-role-of-each-file)
    - [functions.ipynb](#41-functions)
    - [Data-Preprocessing.ipynb](#42-data-preprocessing)
    - [Binary-Task-SVM.ipynb](#43-binary-task-svm)
    - [Binary-Task-CNN.ipynb](#44-binary-task-cnn)
    - [Multiclass-Task-SVM.ipynb](#45-multiclass-task-svm)
    - [Multiclass-Task-CNN.ipynb](#46-multiclass-task-cnn)
    - [Dataset Files.ipynb](#47-dataset-files)
    - [Model files.ipynb](#48-model-files)
    - [Model_Evaulation Files.ipynb](#49-model_evaulation-files)
5. [Running Code to Obtain Same Results](#5-running-code-to-obtain-same-results)

# 1. Overview of Task
The task given is an image classification task regarding Magnetic Resonance Imaging (MRI) images of various brain scans. This is based off the problem on Kaggle regarding [Brain Tumor Classification](https://www.kaggle.com/sartajbhuvaji/brain-tumor-classification-mri). Given an initial dataset of 3000 MRI images with their respective labels containing four unique classes, no_tumor, glioma_tumor, meningioma_tumor and pituitary_tumor. We are to train, validate and test machine learning or deep learning models to solve two problems surrounding the given dataset as follows:

1. Binary Classification
    - Build a classifier to identify whether there is a tumor in the MRI image dataset.
2. Multi-class Classification
    - Build a classifier to identify the type of tumor in each image, meningioma tumor, glioma tumor, pituitary tumor or no tumor.

The task allows any model of any type (Supervised Learning, Unsupervised Learning or Deep Learning) to be used to solve either of the tasks, with the same model allowed to be used for both tasks. However, one non-deep learning model must be implemented.

The models created in this repository are Support Vector Machine (SVM) and Convolutional Neural Network (CNN) based on the AlexNet architecture. Both models were trained for the binary and multi-class classification task.

# 2. Requirements
The `requirements.txt` file present in the repository lists all Python libraries used in the various notebooks and are required to run properly. The necessary requirements can be installed by opening a terminal, changing directory to the workspace containing the repository after cloning and using:
```
pip install -r requirements.txt
```
From the requirements file the libraries used are:
- Keras == 2.3.1
- ipynb == 0.5.1
- matplotlib == 3.4.3
- numpy == 1.21.2
- opencv_python_headless == 4.5.4.60
- pandas == 1.3.4
- scikit_learn == 1.0.1
- tensorflow == 2.1.0
- tqdm == 4.62.3

# 3. Organisation of Repository
The repository is organised as follows:<br/>
The initial commmits were done on the `main` branch, the initial commits contain partially completed code due to the originally created repository having a conflict resolution errror. 

All remaining commits for work done were pushed to the `Coding` branch and documents the progress of the code through the various commits used to solve the two tasks. The fully completed Code was then merged into `main` via Pull Request.

#### Folder and File Organisation Overview
- `Models` contains the saved model files which have been split into binary and multi-class task folders. Within each folder the tuned and untuned SVM models are present with the CNN model saved into the CNN folder `Models\Binary_CNN\CNN` for example. The saved CNN models are in two files, a json file containing the model architecture and the model weights file. This was to reduce file size so it could be uploaded to Github without using LFS.
- `Plots` contains all relevant images of graphs created in the code which were subsequently included in the report. This is to verify that the plots created are the same as the ones in the report.
- `Plots_Original` is a backup for the `Plots` folder which contains all original plots used in the report. Both folders should contain the same images in the original repository presented. In the event of accidental overwritting in `Plots`, refer to `Plots_Original`.
- `dataset` contains all dataset files provided by the module organisers as well as dataset files created during coding/data preprocessing. 
    - All created data files are in the base folder and are saved as pickle files.
    - `label.csv` in the base folder is the starting datset label file, this file came with the original images contained in the `image` folder (3000 MRI images)
    - `Extra_test_dataset` contains the images and label file of the extra test dataset provided by the module organisers one week before the deadline for extra testing.
- All remaining jupyter notebooks in the base repository are files which encompass model training, validation and testing as well as dataset preprocessing and a function file which contains reused functions which can be referenced by the notebooks. Their specific roles are explained in section 4.

# 4. Role of each File
Briefly explains role of the main notebooks and data files created. Full and additional explanations are found in markdown cells and comments in the various notebooks.
### 4.1 functions.ipynb 
Contains reusable functions used in various task notebooks to streamline code. The functions created are:
```
- image_array_resize
- load_dataset
- load_dataset_CNN
- dataset_PCA
- Tuned_SVM_train
- SVM_predictions
```
Their functionality can be found in code comments within `functions.ipynb`
### 4.2 Data-Preprocessing.ipynb
Contains code which carries out various data preprocessing steps on the original MRI image dataset:
1. Exploring the dataset to discover unique labels, number of samples per class and shape of dataframes
2. Carries out relabelling of labels in `label.csv` for both Binary and Multi-class tasks
    - Relabels no_tumor to 0 and any tumor label into (1) for Binary classification and saves file as `Y_Binary_Label.pkl`
    - Relabels no_tumor to 0, glioma_tumor to 1, meningioma_tumor to 2, pituitary_tumor to 3 for multi-class classification and saves file as `Y_Multiclass_Label.pkl`
    - The label files created are saved in the `dataset` folder
3. Obtains MRI image files and stores pixel data into array format. The images are resized and arrays stored separately for SVM and CNN models
    - Images for SVM are resized to 28 * 28, the array is then flattened into a 2D array and saved as `Image_DF_Flat.pkl` in the `dataset` folder.
    - Images for CNN are resized to 50 * 50, the image array for CNN is then converted to a 2D array file and saved as `CNN_Images_2D_DF.pkl` in the `dataset` folder.
4. Finds optimal number of principal components to use for PCA in SVM image data

### 4.3 Binary-Task-SVM.ipynb
Contains code which:
1. Retrives the `Image_DF_Flat.pkl` and `Y_Binary_Label.pkl` files, conducts PCA on the images to extract the most prominent features in the MRI image dataset. 
2. Trains the SVM models
    - Untuned SVM
    - Tuned SVM using `GridSearchCV`
3. Runs predictions with the trained SVM models on a test dataset obtained from `train_test_split` of the original dataset provided (not the extra test dataset)
4. Obtains and prints the `classification_report` and `confusion_matrix` as well as a few other metrics to help assess the out of sample prediction performance of the models.
5. Saves the trained SVM models into the `Models/Binary-Classification` folder
6. Saves any plots created into `Plots` folder.

### 4.4 Binary-Task-CNN.ipynb
Contains code which:
1. Retrives the `CNN_Images_2D_DF.pkl` and `Y_Binary_Label.pkl` files.
2. Generates the CNN model architecture based on AlexNet structure and trains the CNN model
3. Runs predictions with the trained CNN model on a test dataset obtained from `train_test_split` of the original dataset provided (not the extra test dataset)
4. Obtains and prints the `classification_report` and `confusion_matrix` as well as a few other metrics to help assess the out of sample prediction performance of the models.
5. Saves the trained CNN models into the `Models/Binary-Classification/CNN` folder
6. Saves any plots created into `Plots` folder.

### 4.5 Multiclass-Task-SVM.ipynb
Contains code which:
1. Retrives the `Image_DF_Flat.pkl` and `Y_Multiclass_Label.pkl` files, conducts PCA on the images to extract the most prominent features in the MRI image dataset. 
2. Trains the SVM models
    - Untuned SVM
    - Tuned SVM using `GridSearchCV`
3. Runs predictions with the trained SVM models on a test dataset obtained from `train_test_split` of the original dataset provided (not the extra test dataset)
4. Obtains and prints the `classification_report` and `confusion_matrix` as well as a few other metrics to help assess the out of sample prediction performance of the models.
5. Saves the trained SVM models into the `Models/Multiclassification` folder
6. Saves any plots created into `Plots` folder.

### 4.6 Multiclass-Task-CNN.ipynb
Contains code which:
1. Retrives the `CNN_Images_2D_DF.pkl` and `Y_Multiclass_Label.pkl` files.
2. Generates the CNN model architecture based on AlexNet structure and trains the CNN model
3. Runs predictions with the trained CNN model on a test dataset obtained from `train_test_split` of the original dataset provided (not the extra test dataset)
4. Obtains and prints the `classification_report` and `confusion_matrix` as well as a few other metrics to help assess the out of sample prediction performance of the models.
5. Saves the trained CNN models into the `Models/Multiclassification/CNN` folder
6. Saves any plots created into `Plots` folder.

### 4.7 Dataset Files
As mentioned previously, contains all datasets provided by module organisers (Original and extra test dataset) as well as relabeled target label data and preprocessed image arrays for SVM and CNN models

### 4.8 Model files
As mentioned in section 3, these are the saved model files for trained untuned and tuned (Validated) SVM and CNN models for the binary and multi-class tasks. These model files can be reloaded to be used for predictions in separate notebooks, which was done in `Binary-Task-Eval.ipynb` and `Multiclass-Task-Eval.ipynb`

### 4.9 Model_Evaulation Files
- `Binary-Task-Eval.ipynb` contains code which loads trained model files for the **binary classification** task, untuned and tuned SVM and the CNN. It then uses them to run predictions on the `Extra_test_dataset` provided by the module organisers and prints out the `classification_report` and `confusion_matrix` as well as a few other metrics to help assess the out of sample prediction performance of the models.
- `Multiclass-Task-Eval.ipynb` contains code which loads trained model files for the **multiclass classification** task, untuned and tuned SVM and the CNN. It then uses them to run predictions on the `Extra_test_dataset` provided by the module organisers and prints out the `classification_report` and `confusion_matrix` as well as a few other metrics to help assess the out of sample prediction performance of the models.

# 5. Running Code to Obtain Same Results
**NOTE:** All code lines which are used to save models and any plots have been commented out in all the following notebooks, this is to prevent accidental overwriting of the plots with new ones upon code run test as there might be changes in the weights and predictions etc due to slight variance in the model training process (for CNNs, was unable to fix training batches). There is an additional folder `Plots_Original` which is basically a backup for the `Plots` folder which contains all original plots used in the report. Both folders should contain the same images in the original repository presented. In the event of accidental overwritting in `Plots`, refer to `Plots_Original`.

The files to run to obtain all the preprocessed datasets, saved trained models and plots are listed as follows, they are to be run in the order given by the numbering:

1. Data-Preprocessing.ipynb
2. Binary-Task-SVM.ipynb
3. Binary-Task-CNN.ipynb
4. Multiclass-Task-SVM.ipynb
5. Multiclass-Task-CNN.ipynb
6. Binary-Task-Eval.ipynb
7. Multiclass-Task-Eval.ipynb

Dataset preprocessing is done in file 1. Binary Classification Task model training validation and testing on original given dataset is done in 2 and 3. Multi-class Classification Task mdel training validation and testing on original given dataset is done in 4 and 5. Evaluation of Extra Test Dataset using trained models is done in 6 and 7.

Within the Data preprocessing and Binary/Multiclass task model training notebooks (number 1 to 5) the results of the PCA optimal number of components, classification report and confusion matrix as well as other metric used to assess model performance will be printed out after execution ofthe respective code cells. These values were used in conjunction with the plots obtained to be included into the report.

Each notebook file can be run from beginning to end in order of their cells and will produce the repsective results/outputs as per the markdown cells and comments in each notebook describing the process. The files which will be created/overlapped after having run all the notebook files are listed below:
```
- Image_DF_Flat.pkl
- CNN_Images_2D_DF.pkl
- Y_Binary_Label.pkl
- Y_Multiclass_Label.pkl
- Untuned_SVM_model.sav
- Tuned_SVM_model.sav
- Binary_CNN.json
- Binary_CNN_Model_Weights
- Untuned_multiclass_SVM.sav
- Tuned_multiclass_SVM.sav
- Multiclass_CNN.json
- Multiclass_CNN_Model_Weights
- All image files for plots in Plots folder
```


