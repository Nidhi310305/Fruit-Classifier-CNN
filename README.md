# 🍎 Fruit Image Classifier with CNNs

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Keras](https://img.shields.io/badge/Keras-API-red)
![Transfer Learning](https://img.shields.io/badge/Transfer%20Learning-VGG16-success)

## 📌 Project Overview
This repository contains a professional, automated fruit image classification system built using **Convolutional Neural Networks (CNNs)**.

A **custom CNN** and two **transfer learning models (VGG16 and ResNet-50)** were explored for a **10-class fruit classification task**. Based on comprehensive evaluation metrics, **VGG16** emerged as the most accurate and reliable architecture, achieving over 97% accuracy.

The objective of this project is to build robust computer vision models applicable to real-world scenarios such as **agriculture automation, retail checkout systems, and supply chain management**.

---

## 📂 Dataset
- **Dataset Used:** [Fruits 360 Dataset](https://github.com/Horea94/Fruit-Images-Dataset) (10 selected classes)
- Contains labeled images of various fruits captured under controlled conditions.
- Extensive data augmentation (rotation, shifting, flipping, brightness adjustment, zoom) was applied to the training set to prevent overfitting and improve generalization.

---

## 🧠 Model Architecture Focus: VGG16
While multiple architectures were tested, this repository highlights the fine-tuned **VGG16** model due to its superior performance.

### 🔹 VGG16 (Transfer Learning)
- Pre-trained on ImageNet.
- The base convolutional layers were frozen to retain high-level feature extraction capabilities.
- A custom dense classifier block was added (GlobalAveragePooling -> Dense(512) -> Dropout -> Dense(256) -> Dropout -> Softmax(10)).
- **Achieved the best overall performance** across all evaluation metrics, excelling at capturing subtle color and texture differences between fruit classes.

---

## 📊 Evaluation Metrics
The VGG16 model was evaluated using standard classification metrics:

| Metric       | VGG16 Performance |
|-------------|-------------------|
| **Accuracy**    | **97.23%**        |
| **Precision**   | **97.62%**        |
| **Recall**      | **96.95%**        |
| **F1-Score**    | **96.98%**        |

*Note: The Custom CNN (94.68% accuracy) and ResNet-50 (94.35% accuracy) implementations are also available for comparison.*

---

## ⚙️ Repository Structure

```
Fruit-Classifier-CNN/
│
├── src/
│   ├── __init__.py
│   ├── config.py           # Hyperparameters and paths
│   ├── data_loader.py      # Image generators and augmentation
│   ├── models.py           # Model definitions (VGG16)
│   ├── train.py            # Training pipeline with callbacks
│   └── evaluate.py         # Testing and metrics generation
│
├── notebooks/
│   └── Capstone_Project.ipynb  # Original research notebook
│
├── requirements.txt        # Python dependencies
└── README.md
```

## 🚀 Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/Nidhi310305/Fruit-Classifier-CNN.git
cd Fruit-Classifier-CNN
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Ensure dataset is in place**
Place your dataset inside `Fruit_360_Dataset/Training` and `Fruit_360_Dataset/Testing` at the root of the project, or update `src/config.py` to point to your dataset location.

4. **Train the model**
```bash
python -m src.train
```

5. **Evaluate the model**
```bash
python -m src.evaluate
```
