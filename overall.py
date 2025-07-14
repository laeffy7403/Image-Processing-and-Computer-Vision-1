import os
import cv2
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# ========== CONFIG ==========
IMAGE_SIZE = (64, 64)  # resize for faster processing
DATA_DIR = "dataset"
LABELS = {"cat": 0, "dog": 1}

# ============================

def load_images(data_dir, labels_dict):
    """
    Load and preprocess all images from the directory.
    Returns:
        - features: list of flattened or shaped images
        - labels: list of integer labels
    """
    features = []
    labels = []
    print("Loading and preprocessing images...")

    for label_name, label_val in labels_dict.items():
        path = os.path.join(data_dir, label_name)
        for img_name in os.listdir(path):
            if img_name.endswith(".jpg"):
                img_path = os.path.join(path, img_name)
                img = cv2.imread(img_path)
                img = cv2.resize(img, IMAGE_SIZE)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # for traditional ML
                img_flat = img.flatten()
                features.append(img_flat)
                labels.append(label_val)

    print(f"Loaded {len(features)} images.")
    return np.array(features), np.array(labels)

def load_images_cnn(data_dir, labels_dict):
    """
    Same as load_images, but returns 3D images for CNN.
    """
    features = []
    labels = []
    for label_name, label_val in labels_dict.items():
        path = os.path.join(data_dir, label_name)
        for img_name in os.listdir(path):
            if img_name.endswith(".jpg"):
                img_path = os.path.join(path, img_name)
                img = cv2.imread(img_path)
                img = cv2.resize(img, IMAGE_SIZE)
                img = img / 255.0  # Normalize to [0, 1]
                features.append(img)
                labels.append(label_val)
    return np.array(features), np.array(labels)

# ================= RANDOM FOREST + SVM =================
def train_ml_models(X, y):
    print("\nTraining traditional ML models (Random Forest and SVM)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    start = time.time()
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    print("Random Forest:\n", classification_report(y_test, rf_preds))
    print("RF Training Time:", time.time() - start)

    start = time.time()
    svm = SVC(kernel='linear')
    svm.fit(X_train, y_train)
    svm_preds = svm.predict(X_test)
    print("SVM:\n", classification_report(y_test, svm_preds))
    print("SVM Training Time:", time.time() - start)

# ====================== CNN ==========================
def build_cnn(input_shape, num_classes):
    """
    Build a simple CNN using Keras Sequential API.
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_cnn(X, y):
    print("\nTraining CNN model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)

    model = build_cnn(input_shape=X_train.shape[1:], num_classes=y_train_cat.shape[1])

    start = time.time()
    history = model.fit(X_train, y_train_cat, epochs=10, batch_size=32, validation_split=0.1, verbose=1)
    print("CNN Training Time:", time.time() - start)

    test_loss, test_acc = model.evaluate(X_test, y_test_cat)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")

    # Optional: plot training loss/accuracy
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='val')
    plt.legend()
    plt.title("CNN Accuracy Over Epochs")
    plt.show()

# ==================== MAIN ============================
if __name__ == "__main__":
    start_all = time.time()

    # Load and train Random Forest / SVM
    X_ml, y_ml = load_images(DATA_DIR, LABELS)
    train_ml_models(X_ml, y_ml)

    # Load and train CNN
    X_cnn, y_cnn = load_images_cnn(DATA_DIR, LABELS)
    train_cnn(X_cnn, y_cnn)

    print(f"\nTotal pipeline completed in {time.time() - start_all:.2f} seconds.")
