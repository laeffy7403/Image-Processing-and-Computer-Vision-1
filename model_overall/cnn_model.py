import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

def build_cnn(input_shape, num_classes):
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
    print(f"CNN Training Time: {time.time() - start:.2f}s")

    test_loss, test_acc = model.evaluate(X_test, y_test_cat)
    print(f"CNN Test Accuracy: {test_acc * 100:.2f}%")

    # Plot Accuracy
    # plt.plot(history.history['accuracy'], label='train')
    # plt.plot(history.history['val_accuracy'], label='val')
    # plt.legend()
    # plt.title("CNN Accuracy Over Epochs")
    # plt.show()
