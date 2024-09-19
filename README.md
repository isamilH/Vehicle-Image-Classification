# Vehicle-Image-Classification


Overview: This dataset is designed for vehicle classification tasks and contains a total of 5,600 images distributed across seven categories. Each category represents a different type of vehicle.

Structure:

    Main Folder: Vehicles
    Subfolders:
        Auto Rickshaws (800 images)
        Bikes (800 images)
        Cars (800 images)
        Motorcycles (800 images)
        Planes (800 images)
        Ships (800 images)
        Trains (800 images)

## Introduction

The goal of this project is to classify vehicle images into predefined categories using machine learning techniques. The dataset used includes labeled images of vehicles from various classes.
Data Preprocessing

The preprocessing steps involve:

    Loading the dataset and organizing it into training, validation, and test sets.
    Performing data augmentation to increase the size and diversity of the dataset.
    Normalizing the images to ensure consistency in pixel values.

##Model Architecture:
The model used for image classification is a convolutional neural network (CNN). It consists of several convolutional layers followed by pooling layers to extract relevant features from the images.

1. Input Layer: Resized vehicle images.
2. Convolutional Layers: Feature extraction layers with filters applied to detect edges, shapes, etc.
3. Pooling Layers: Reducing the spatial dimensions of the extracted features.
4. Fully Connected Layers: Combining the features to predict the vehicle class.

## Model Training
The model was trained using a categorical cross-entropy loss function and an optimizer such as Adam or SGD. Early stopping and learning rate adjustments were applied to prevent overfitting and optimize training speed.

## Evaluation and Results
The trained model was evaluated using accuracy. The results showed 92% accuracy on the test set. 

## Conclusion
This project demonstrates the effectiveness of CNNs for vehicle image classification. Further improvements could be achieved by using transfer learning from pre-trained models such as ResNet or VGG, or by experimenting with different hyperparameters and architectures.
