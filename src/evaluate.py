import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from src.config import CLASS_NAMES, FIGURES_PATH, MODEL_SAVE_PATH
from src.data_loader import get_data_generators

def evaluate_model(model_path=MODEL_SAVE_PATH):
    """
    Loads a trained model and evaluates it on the test dataset.
    Generates a confusion matrix and classification report.
    """
    print(f"Loading model from {model_path}...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please train the model first.")

    model = tf.keras.models.load_model(model_path)
    _, _, test_gen = get_data_generators()

    print("Evaluating on test data...")
    test_gen.reset()
    test_loss, test_accuracy = model.evaluate(test_gen, verbose=1)
    
    print(f"\nTest Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Get predictions
    test_gen.reset()
    predictions = model.predict(test_gen, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_gen.classes

    # Clean class names for plotting
    clean_class_names = [name.replace(' 1', '') for name in CLASS_NAMES]

    # Confusion Matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    
    os.makedirs(FIGURES_PATH, exist_ok=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=clean_class_names,
                yticklabels=clean_class_names)
    plt.title(f'VGG16 - Confusion Matrix\nTest Accuracy: {test_accuracy:.3f}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    cm_path = os.path.join(FIGURES_PATH, 'vgg16_confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved as: {cm_path}")

    # Classification Report
    print("\nClassification Report:")
    report = classification_report(true_classes, predicted_classes, target_names=clean_class_names)
    print(report)

if __name__ == "__main__":
    evaluate_model()
