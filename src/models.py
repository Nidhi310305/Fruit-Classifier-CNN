import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from src.config import IMG_SIZE, NUM_CLASSES, LEARNING_RATE

def build_vgg16_model():
    """
    Builds and compiles a fine-tuned VGG16 model for fruit classification.
    """
    print("Building VGG16 Fine-tuned Model...")

    # Load VGG16 with ImageNet pre-trained weights, excluding the top classifier
    base_vgg16 = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    # Freeze the base model to retain pre-trained features
    base_vgg16.trainable = False

    # Add custom classifier on top
    model = models.Sequential([
        base_vgg16,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f"VGG16 created with {model.count_params():,} parameters.")
    return model
