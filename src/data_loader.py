from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.config import TRAIN_DIR, TEST_DIR, IMG_SIZE, BATCH_SIZE
import os

def get_data_generators():
    """
    Creates and returns training, validation, and testing data generators.
    Includes data augmentation for the training set.
    """
    print("Setting up Data Generators...")

    # Training data with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,                    
        rotation_range=20,                 
        width_shift_range=0.2,             
        height_shift_range=0.2,            
        horizontal_flip=True,              
        brightness_range=[0.8, 1.2],      
        zoom_range=0.2,                    
        fill_mode='nearest',               
        validation_split=0.2               
    )

    # Test data (no augmentation, only normalization)
    test_datagen = ImageDataGenerator(rescale=1./255)

    if not os.path.exists(TRAIN_DIR) or not os.path.exists(TEST_DIR):
        raise FileNotFoundError(f"Dataset directories not found. Please ensure {TRAIN_DIR} and {TEST_DIR} exist.")

    # Create generators
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),    
        batch_size=BATCH_SIZE,               
        class_mode='categorical',            
        subset='training',                   
        shuffle=True,                        
        seed=42                              
    )

    validation_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,                           
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',                 
        shuffle=False,                       
        seed=42
    )

    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False                        
    )

    return train_generator, validation_generator, test_generator
