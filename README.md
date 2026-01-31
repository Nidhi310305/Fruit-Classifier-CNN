# Fruit-Classifier-CNN
# 🍎 Fruit Image Classifier with CNNs

## 📌 Project Overview
This project focuses on building an automated fruit image classification system using **Convolutional Neural Networks (CNNs)**.  
A **custom CNN** and two **transfer learning models (VGG16 and ResNet-50)** were implemented and compared for a **10-class fruit classification task**.

The objective is to evaluate model performance and identify the most reliable architecture for real-world applications in **agriculture and supply chain automation**.

---

## 📂 Dataset
- **Dataset Used:** Fruits360  
- Contains labeled images of various fruits captured under controlled conditions  
- Dataset was split into training, validation, and test sets  
- Images were preprocessed and augmented to improve generalization

---

## 🧠 Model Architectures
Three different models were trained and evaluated:

### 🔹 Custom CNN
- Built from scratch using convolutional, pooling, and dense layers
- Lightweight architecture with competitive performance
- Suitable for environments with limited computational resources

### 🔹 VGG16 (Transfer Learning)
- Pre-trained on ImageNet
- Fine-tuned for fruit classification
- Achieved the best overall performance across all evaluation metrics

### 🔹 ResNet-50 (Transfer Learning)
- Deep residual network with skip connections
- Fine-tuned for multi-class fruit classification
- Performed competitively but slightly below VGG16

---

## 📊 Evaluation Metrics
The models were evaluated using the following metrics:
- **Accuracy**
- **Precision (Macro)**
- **Recall (Macro)**
- **F1-Score (Macro)**
- **Confusion Matrix**

### 🔎 Performance Comparison

| Metric       | Custom CNN | VGG16 | ResNet-50 |
|-------------|------------|--------|-----------|
| Accuracy    | 94.68%     | 97.23% | 94.35%    |
| Precision   | 95.99%     | 97.62% | 95.30%    |
| Recall      | 94.16%     | 96.95% | 93.80%    |
| F1-Score    | 93.86%     | 96.98% | 93.47%    |

---

## 📈 Results & Observations
- **VGG16** consistently outperformed other models across all metrics
- **Custom CNN** provided strong performance with lower computational cost
- **ResNet-50** achieved respectable results but slightly lagged behind

Confusion matrices and comparative plots were used to analyze per-class performance and model trade-offs.

---

## ⚙️ Setup Instructions

```bash
git clone https://github.com/Nidhi310305/Fruit-Classifier-CNN
Fruit-Classifier-CNN
pip install -r requirements.txt
