import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import accuracy_score, classification_report
import sys

dataset_path = sys.argv[1]               
target_column = sys.argv[2]

try:
    df = pd.read_csv("./nba_stats.csv")  
    le = LabelEncoder()
    df['Pos_encoded'] = le.fit_transform(df['Pos'])

    unwanted_columns = ['Pos', 'Age', 'FG%', '3P%', '2P%', 'eFG%', 'FT%']
    df.drop(unwanted_columns, axis=1, inplace=True)

    X = df.drop(columns=['Pos_encoded'])
    y = df['Pos_encoded']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

    # Scale the features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    gnb = ComplementNB()
    gnb.fit(X_train_scaled, y_train)

    y_train_pred = gnb.predict(X_train_scaled)
    y_val_pred = gnb.predict(X_val_scaled)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    val_accuracy = accuracy_score(y_val, y_val_pred)

    val_report = classification_report(y_val, y_val_pred, output_dict=True)
    precision = val_report["weighted avg"]["precision"] * 100
    recall = val_report["weighted avg"]["recall"] * 100
    f1_score = val_report["weighted avg"]["f1-score"] * 100

    # Print the results
    print(f"Training Set Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Validation Set Accuracy: {val_accuracy * 100:.2f}%")
    print(f"Precision: {precision:.2f}%")
    print(f"Recall: {recall:.2f}%")
    print(f"F1-Score: {f1_score:.2f}%")

    cm1 = pd.crosstab(y_train, y_train_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
    cm2 = pd.crosstab(y_val, y_val_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)

    print("\nTraining Confusion Matrix:\n", cm1)
    print("\nValidation Confusion Matrix:\n", cm2)

except FileNotFoundError:
    print("Error: The file was not found. Please check the file path and try again.")
except Exception as e:
    print(f"An error occurred: {e}")

# Calculate Testing Accuracy (dummy_test.csv)
try:
    test_df = pd.read_csv("./dummy_test.csv")  

    test_df['Pos_encoded'] = le.transform(test_df['Pos'])
    test_df.drop(unwanted_columns, axis=1, inplace=True, errors='ignore')

    X_test = test_df.drop(columns=['Pos_encoded'])
    y_test = test_df['Pos_encoded']

    X_test_scaled = scaler.transform(X_test)
    y_test_pred = gnb.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print(f"Test Set Accuracy: {test_accuracy * 100:.2f}%")

    cm3 = pd.crosstab(y_test, y_test_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
    print("\nTesting Confusion Matrix:\n", cm3)

except FileNotFoundError:
    print("Error: The testing file was not found. Please check the file path and try again.")
except Exception as e:
    print(f"An error occurred: {e}")

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
cv_scores = cross_val_score(ComplementNB(), X_train_scaled, y_train, cv=skf, scoring='accuracy')

# Print Cross-Validation Results
print("\nCross-Validation Accuracy Scores:")
for i, score in enumerate(cv_scores, 1):
    print(f"Fold {i}: {score * 100:.2f}%")

average_cv_accuracy = cv_scores.mean()
print(f"\nAverage Cross-Validation Accuracy: {average_cv_accuracy * 100:.2f}%")