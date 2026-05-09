import os

# --- Dataset Settings ---
DATASET_PATH = "Fruit_360_Dataset"
TRAIN_DIR = os.path.join(DATASET_PATH, "Training")
TEST_DIR = os.path.join(DATASET_PATH, "Testing")

# --- Model & Training Parameters ---
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 10
EPOCHS = 15
LEARNING_RATE = 0.001

CLASS_NAMES = [
    'Apple Red Delicious 1', 'Banana 1', 'Cherry 1', 'Grape Blue 1', 
    'Lemon 1', 'Orange 1', 'Peach 1', 'Pear 1', 'Pineapple 1', 'Strawberry 1'
]

# --- Paths ---
MODEL_SAVE_PATH = "models/best_vgg16.keras"
FIGURES_PATH = "evaluation_figures/"
