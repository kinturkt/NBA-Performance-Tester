import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import sys

dataset_path = sys.argv[1]               
target_column = sys.argv[2]

try:
    df = pd.read_csv('./nba_stats.csv')

    # Apply Label Encoding
    le = LabelEncoder()
    df['Pos_encoded'] = le.fit_transform(df['Pos'])

    unwanted_columns = ['Pos', 'Age', 'FG%', '3P%', '2P%', 'TRB', 'FT%']
    df.drop(unwanted_columns, axis=1, inplace=True)

    X = df.drop(columns=['Pos_encoded'])
    y = df['Pos_encoded']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=0, stratify=y)

    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    svm = SVC(kernel='linear', C=4, gamma='scale', random_state=0)
    svm.fit(X_train_scaled, y_train)

    y_train_pred = svm.predict(X_train_scaled)
    y_val_pred = svm.predict(X_val_scaled)

    train_acc = accuracy_score(y_train, y_train_pred)
    valid_acc = accuracy_score(y_val, y_val_pred)

    test_df = pd.read_csv("./dummy_test.csv")  
    test_df['Pos_encoded'] = le.transform(test_df['Pos'])
    test_df.drop(unwanted_columns, axis=1, inplace=True, errors='ignore')

    X_test = test_df.drop(columns=['Pos_encoded'])
    y_test = test_df['Pos_encoded']

    X_test_scaled = scaler.transform(X_test)
    y_test_pred = svm.predict(X_test_scaled)

    test_accuracy = accuracy_score(y_test, y_test_pred)

    # Generate classification report
    val_report = classification_report(y_val, y_val_pred, output_dict=True)
    precision = val_report["weighted avg"]["precision"] * 100
    recall = val_report["weighted avg"]["recall"] * 100
    f1_score = val_report["weighted avg"]["f1-score"] * 100

    # Print Results
    print(f"Training Set Accuracy: {train_acc * 100:.2f}%")
    print(f"Validation Set Accuracy: {valid_acc * 100:.2f}%")
    print(f"Test Set Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Precision: {precision:.2f}%")
    print(f"Recall: {recall:.2f}%")
    print(f"F1-Score: {f1_score:.2f}%")

    cm1 = pd.crosstab(y_train, y_train_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
    cm2 = pd.crosstab(y_val, y_val_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
    cm3 = pd.crosstab(y_test, y_test_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)

    print("\nTraining Confusion Matrix:\n", cm1)
    print("\nValidation Confusion Matrix:\n", cm2)
    print("\nTesting Confusion Matrix:\n", cm3)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    cv_scores = cross_val_score(SVC(kernel='linear', gamma='scale'), X_train_scaled, y_train, cv=skf, scoring='accuracy')

    print("\nCross-Validation Accuracy Scores:")
    for i, score in enumerate(cv_scores, 1):
        print(f"Fold {i}: {score * 100:.2f}%")

    average_cv_accuracy = cv_scores.mean()
    print(f"\nAverage Cross-Validation Accuracy: {average_cv_accuracy * 100:.2f}%")

except FileNotFoundError:
    print("Error: The file was not found. Please check the file path and try again.")
except Exception as e:
    print(f"An error occurred: {e}")