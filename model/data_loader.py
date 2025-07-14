import os
import cv2
import numpy as np

IMAGE_SIZE = (64, 64)

def load_images(data_dir, labels_dict):
    features = []
    labels = []
    print("Loading grayscale images for ML...")

    for label_name, label_val in labels_dict.items():
        path = os.path.join(data_dir, label_name)
        if not os.path.exists(path): continue
        for img_name in os.listdir(path):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(path, img_name)
                img = cv2.imread(img_path)
                if img is None: continue
                img = cv2.resize(img, IMAGE_SIZE)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img_flat = img.flatten()
                features.append(img_flat)
                labels.append(label_val)

    print(f"Loaded {len(features)} grayscale images.")
    return np.array(features), np.array(labels)

def load_images_cnn(data_dir, labels_dict):
    features = []
    labels = []
    print("Loading RGB images for CNN...")

    for label_name, label_val in labels_dict.items():
        path = os.path.join(data_dir, label_name)
        if not os.path.exists(path): continue
        for img_name in os.listdir(path):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(path, img_name)
                img = cv2.imread(img_path)
                if img is None: continue
                img = cv2.resize(img, IMAGE_SIZE)
                img = img / 255.0
                features.append(img)
                labels.append(label_val)

    print(f"Loaded {len(features)} RGB images.")
    return np.array(features), np.array(labels)
