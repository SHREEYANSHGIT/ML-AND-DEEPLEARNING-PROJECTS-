Cat vs Dog Classification using CNN
========================================

This mini project focuses on classifying images of cats and dogs using a Convolutional Neural Network (CNN) built with TensorFlow and Keras. The model is trained on 3000+ images and achieves 92%+ accuracy, making it capable of predicting any cat or dog image with high reliability.

----------------------------------------
Project Overview
----------------------------------------
- Used a dataset of 3000+ labeled cat and dog images
- Applied image preprocessing and augmentation
- Built a custom CNN architecture (Conv2D, MaxPooling, Dropout, Dense layers)
- Trained using Adam optimizer and categorical cross-entropy
- Achieved 92%+ test accuracy
- Added a prediction script for custom image classification

----------------------------------------
Tech Stack
----------------------------------------
Python | TensorFlow/Keras | NumPy | Pandas | Matplotlib | OpenCV | Jupyter Notebook

----------------------------------------
Model Performance
----------------------------------------
- Accuracy: 92%+
- Works across different breeds, lighting conditions, and angles
- Robust predictions on real-world images

----------------------------------------
Model Architecture
----------------------------------------
- Multiple Convolutional + MaxPooling layers
- Dropout for overfitting reduction
- Dense layers with ReLU activation
- Softmax output layer for 2-class classification

----------------------------------------
Files Included
----------------------------------------
- cat_vs_dog.ipynb — Main notebook
- dataset/ — Cat & dog dataset (or download link)
- prediction.py — Script for predicting new images
- requirements.txt — Dependencies

----------------------------------------
How to Run
----------------------------------------
pip install -r requirements.txt
jupyter notebook "cat_vs_dog.ipynb"

To predict your own image:
python prediction.py --image your_image.jpg

----------------------------------------
Author
----------------------------------------
Shreeyansh Asati
