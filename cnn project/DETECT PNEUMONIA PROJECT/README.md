## 🫁 Pneumonia Detection Using CNN & Transfer Learning
A Deep Learning Approach for Medical Image Classification

This project implements an automated system to classify chest X-ray images into NORMAL and PNEUMONIA using a combination of Convolutional Neural Networks (CNN) and Transfer Learning (MobileNetV2). The model is optimized using RMSprop with a carefully tuned learning rate, making it both stable and accurate for medical imaging tasks.

## ⭐ Project Overview

Chest X-ray interpretation is a critical diagnostic tool, but manual inspection is time-consuming and prone to human error. This project aims to automate pneumonia detection by training a deep learning model that generalizes well across diverse chest radiographs.

The model leverages MobileNetV2 for feature extraction and adds a custom CNN classifier trained specifically for pneumonia detection. It includes:

Efficient preprocessing pipeline

Handling of class imbalance

Evaluation metrics and graph visualizations

Prediction on user-uploaded images

This makes it suitable for academic research, clinical prototyping, and machine learning portfolios.

## 📂 Dataset Structure
-  dataset link : https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Ensure the dataset is organized as follows:

- chest_xray/
- │── train/
- │    ├── NORMAL/
- │    ├── PNEUMONIA/
- │── val/
- │    ├── NORMAL/
- │    ├── PNEUMONIA/
- │── test/
     - ├── NORMAL/
     - ├── PNEUMONIA/


This format allows TensorFlow to automatically label images and load them efficiently.

## 🧠 Model Architecture

The model consists of two main components:

🔹 1. Transfer Learning Backbone (MobileNetV2)

Pretrained on ImageNet

Used as a fixed feature extractor

include_top=False

Frozen during initial training

Input size: 224 × 224 × 3

🔹 2. Custom CNN Classification Head

Global Average Pooling

Dropout for regularization

Sigmoid output for binary classification

This architecture ensures both performance and efficiency, achieving high accuracy with limited data.

## ⚙️ Optimization Strategy (RMSprop)

The model uses the RMSprop optimizer with tuned hyperparameters:

optimizer = tf.keras.optimizers.RMSprop(
    learning_rate=1e-4,
    rho=0.9
)

## ✔ Why RMSprop?

Adapts learning rate individually for each parameter

Excellent stability for medical image data

Works effectively with Transfer Learning

rho=0.9 smooths gradient updates and reduces oscillation

This optimizer helps prevent overfitting and improves convergence.

## ⚖️ Handling Class Imbalance

Medical datasets typically contain more pneumonia cases than normal cases. To avoid bias, we compute class weights:

class_weight = {0: weight_normal, 1: weight_pneumonia}


This ensures balanced learning, improving recall for minority classes like NORMAL.

## 📊 Model Evaluation

The project provides multiple evaluation methods:

Training & validation accuracy curves

Training & validation loss curves

Confusion Matrix

Classification Report (Precision, Recall, F1-score)

Sample predictions with images

These tools give full visibility into the model’s performance.

## 🧪 Predicting a New Image

Users can classify any external chest X-ray image:

new_image_path = input("Enter full image path: ").strip().replace("\\", "/")
predict_new_image(model, new_image_path)


## The system will:

Load the image

Preprocess it

Predict NORMAL or PNEUMONIA

Show the image with prediction and probability

## 📌 Conclusion

This project demonstrates the power of combining CNNs with Transfer Learning for medical image classification. By using MobileNetV2 features and optimizing with RMSprop (learning_rate=1e-4, rho=0.9), the model achieves strong and stable performance with minimal computational cost.

The workflow includes:

Balanced training

Robust evaluation

User-driven prediction

This project provides a solid foundation for building real-world diagnostic tools and can be extended further with:

Grad-CAM heatmaps

Web deployment

Mobile app integration

Training on larger clinical datasets

## 📈 Model Performance

Accuracy: 90+ %

## 🏅 Author

Shreeyansh Asati 

## 📁 Files
- google colab link : https://colab.research.google.com/drive/1PQqsICM1DhzfVOD2r32c-kT43NRvtkhq?usp=sharing
- `final pneumonia prediction model.ipynb` — main notebook
- `final pneumonia prediction model_.py` — main python file 
- `requirements.txt` — dependencies list

## 🧾 How to Run
```bash
pip install -r requirements.txt
jupyter notebook final pneumonia prediction model.ipynb
