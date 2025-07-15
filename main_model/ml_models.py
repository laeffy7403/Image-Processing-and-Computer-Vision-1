import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def train_ml_models(X, y):
    print("\nTraining traditional ML models (Random Forest and SVM)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Random Forest
    start = time.time()
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    print("Random Forest:\n", classification_report(y_test, rf_preds))
    print(f"RF Training Time: {time.time() - start:.2f}s")

    # SVM
    start = time.time()
    svm = SVC(kernel='linear')
    svm.fit(X_train, y_train)
    svm_preds = svm.predict(X_test)
    print("SVM:\n", classification_report(y_test, svm_preds))
    print(f"SVM Training Time: {time.time() - start:.2f}s")
