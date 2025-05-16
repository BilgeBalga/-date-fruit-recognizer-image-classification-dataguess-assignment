# Date Fruit Variety Image Classification Project

This project involves developing a deep learning model using TensorFlow and Keras to classify different varieties of date fruits. The model uses Convolutional Neural Networks (CNNs) for image classification and can identify 9 different date varieties with high accuracy.

## Project Overview

This study aims to develop an image classification model using date fruit images captured in a controlled environment. The project has potential applications in the agricultural industry, such as quality control, automatic classification, and product tracking.

## Dataset

The dataset used in this project is the **Date Fruit Image Dataset in Controlled Environment**, available on Kaggle:  
[https://www.kaggle.com/datasets/wadhasnalhamdan/date-fruit-image-dataset-in-controlled-environment](https://www.kaggle.com/datasets/wadhasnalhamdan/date-fruit-image-dataset-in-controlled-environment)

It includes images of 9 different date varieties:

- Ajwa  
- Medjool  
- Nabtat Ali  
- Shaishe  
- Sugaey  
- Galaxy  
- Meneifi  
- Rutab  
- Sokari

## Requirements

The following libraries are required to run the project:

- `numpy`  
- `pandas`  
- `matplotlib`  
- `tensorflow>=2.0`  
- `scikit-learn`  
- `seaborn`  
- `pillow`

## Model Architecture

The project uses a CNN architecture consisting of four convolutional blocks:

1. Convolutional layer with 32 filters (3x3) + ReLU activation + 2x2 max pooling  
2. Convolutional layer with 64 filters (3x3) + ReLU activation + 2x2 max pooling  
3. Convolutional layer with 128 filters (3x3) + ReLU activation + 2x2 max pooling  
4. Convolutional layer with 256 filters (3x3) + ReLU activation + 2x2 max pooling  
5. Flatten layer  
6. Fully connected layer with 128 neurons + ReLU activation + 50% Dropout  
7. Output layer with 9 neurons + Softmax activation  

## Training Process

The model was trained using the following approach:

- The data was split into 80% training and 20% test. The test set was further split equally for testing and validation  
- Images were resized to 224x224 pixels and normalized between 0–1  
- Adam optimizer and Categorical Cross-Entropy loss function were used  
- Techniques such as **early stopping**, **learning rate reduction**, and **dropout** were applied to prevent overfitting  
- The model was trained for 20 epochs, but early stopping ensured the best-performing model was saved  

## 📊 Model Performance

The model evaluation metrics are as follows:

- **Test Accuracy**: 85.54%  
- **Class-wise Performance**: Precision, recall, and F1-score were calculated for each class  
- **Confusion Matrix**: Used to visualize class-wise misclassifications  
- **ROC Curves**: ROC curves and AUC values were calculated for each class  
- **Precision-Recall Curves**: Class-wise precision-recall performance was evaluated  

## Industrial Application Potential

This model has potential applications in industrial settings, including:

1. **Automated Quality Control**: Can be used in image-based quality control systems to assess date variety and quality  
2. **Smart Agriculture**: Useful in post-harvest processes for automatically sorting and classifying date varieties, reducing labor costs and increasing efficiency  
3. **Supply Chain Management**: Can help track and verify agricultural products like dates across the supply chain  
4. **Mobile Applications**: Can be used in apps for farmers and consumers to identify date varieties  

## Future Improvements

To enhance model performance and usability, the following improvements can be considered:

- Improve performance using transfer learning with pre-trained models (e.g., VGG16, ResNet50)  
- Enhance model generalization using data augmentation techniques  
- Extend the model to recognize more date varieties  
- Optimize the model for low-resource devices (e.g., using TensorFlow Lite)  
- Adapt the model to recognize dates at different ripening stages  

## Note: About the Model File

The trained model file (.h5) could not be uploaded to GitHub due to size limitations. However, it can be recreated by running the code in the repository. Alternatively, you can access the trained model via the Drive link provided here: [Model](https://drive.google.com/drive/folders/1w0_1MSat9HHM80OrcV9xesapT4JuqtTN?usp=sharing)
