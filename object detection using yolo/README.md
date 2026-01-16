# 🚀 YOLOv8 Custom Object Detection using Ultralytics

**👤 Author:** Shreeyansh Asati

**🔗 LinkedIn:** https://www.linkedin.com/in/shreeyansh-asati-18shreey/

**📁 Project Type:** Computer Vision | Object Detection | Deep Learning

---

# 📌 PROJECT OVERVIEW

This repository contains my first end-to-end custom object detection project built using YOLOv8 (You Only Look Once) with the Ultralytics framework. The objective of this project was to move beyond theoretical learning and gain practical, hands-on experience by creating and training an object detection model on a custom dataset.

Recently, I started learning about YOLO and modern object detection techniques. To strengthen my understanding through real-world implementation, I decided to build this project from scratch, covering the complete workflow — from data collection and annotation to model training and evaluation.

---

# 🧠 PROBLEM STATEMENT

Object detection is a core problem in computer vision where the goal is to identify and localize multiple objects within an image. This project focuses on detecting everyday objects in real-world scenes using bounding boxes and class labels.

---
# 📌 PROJECT SHOWCASE

<img width="732" height="581" alt="Screenshot 2026-01-16 144449" src="https://github.com/user-attachments/assets/8c7bf3e3-e033-4590-97e3-49ae3161f9db" />


https://github.com/user-attachments/assets/2592c72a-80eb-41c8-b7ad-41c7c1791b86




# 📂 DATASET DETAILS & ANNOTATION

• **Total Images:** 108

• **Number of Classes:** 5

**Class Labels:**

[0 – bottle, 1 – person ,2 – phone , 3 – facewash , 4 – pen]

All images were manually annotated using **LabelImg**. Bounding boxes were created in **YOLO format**, which helped me understand annotation accuracy, class indexing, and how annotation quality directly affects model performance.

The dataset was structured following YOLO best practices, with separate folders for training and validation images and labels.

---

# 🏋️ MODEL TRAINING DETAILS

• **Model Architecture:** YOLOv8 (Ultralytics)

• **Task:** Object Detection

• **Framework:** PyTorch

• **Image Size:** 640 × 640

• **Epochs:** 100


Pretrained YOLOv8 weights were used as a starting point, allowing the model to learn faster and generalize better on the custom dataset.

---

# 📊 MODEL EVALUATION METRICS

The model was evaluated using standard object detection metrics:

• **Precision:** ~79%

• **Recall:** 100%

• **mAP@0.5:** 0.995

• **mAP@0.5–0.95:** ~0.79


These results indicate that the model is highly effective at detecting objects with minimal missed detections and strong performance across multiple IoU thresholds.

---

# 🔍 KEY LEARNINGS

✔ Understanding YOLO architecture and training workflow

✔ Creating and managing a custom dataset

✔ Manual bounding box annotation using **LabelImg**

✔ Interpreting **precision, recall, and mAP** metrics

✔ Importance of data quality and class balance

✔ Debugging common training and dataset errors

---

# 🛠 TECHNOLOGY STACK

• **Python** 🐍

• **YOLOv8 (Ultralytics)**

• **PyTorch**

• **OpenCV**

• **LabelImg**


---

# 📁 PROJECT STRUCTURE

• `dataset/` – Custom images, labels and data.yaml

• `runs/` – Training outputs, metrics, and weights

---

# 🎯 CONCLUSION

This project represents my first practical implementation of object detection using YOLOv8. It helped me build a strong foundation in computer vision and develop confidence in working with real-world ML pipelines. The experience gained through this project has motivated me to further explore advanced topics such as instance segmentation, model optimization, and deployment strategies.

---

**⭐ If you find this project useful or interesting, feel free to star the repository!**
