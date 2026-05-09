import os
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from src.config import EPOCHS, MODEL_SAVE_PATH
from src.data_loader import get_data_generators
from src.models import build_vgg16_model

def train_model():
    """
    Executes the training pipeline for the VGG16 model.
    """
    print("Starting Training Pipeline...")

    # 1. Load Data
    train_gen, val_gen, test_gen = get_data_generators()

    # 2. Build Model
    model = build_vgg16_model()

    # 3. Setup Callbacks
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
        ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, mode='max')
    ]

    # 4. Train Model
    print(f"Training for {EPOCHS} epochs...")
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )

    print(f"Training Complete! Best model saved to {MODEL_SAVE_PATH}")
    return model, history

if __name__ == "__main__":
    train_model()
