import time
from model.data_loader import load_images, load_images_cnn
from model.ml_models import train_ml_models
from model.cnn_model import train_cnn


# === CONFIG ===
DATA_DIR = "dataset"
LABELS = {"classA": 0, "classB": 1}
# ==============

if __name__ == "__main__":
    start_all = time.time()

    # Load and train traditional ML models
    X_ml, y_ml = load_images(DATA_DIR, LABELS)
    train_ml_models(X_ml, y_ml)

    # Load and train CNN
    X_cnn, y_cnn = load_images_cnn(DATA_DIR, LABELS)
    train_cnn(X_cnn, y_cnn)

    print(f"\n✅ Total pipeline completed in {time.time() - start_all:.2f} seconds.")
