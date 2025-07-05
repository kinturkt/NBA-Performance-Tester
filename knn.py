import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import sys

param = float(sys.argv[1])
dataset_path = sys.argv[2]
target_column = sys.argv[3]  

dataset_path = './nba_stats.csv'  # Update the path if necessary during the runtime of the code
df = pd.read_csv(dataset_path)
df.shape

le = LabelEncoder()
df['Pos_encoded'] = le.fit_transform(df['Pos'])

df.drop(['Pos', 'Age', 'FG%', '3P%', '2P%', 'eFG%', 'FT%'], axis=1, inplace=True)

X = df.drop(columns=['Pos_encoded'])
y = df['Pos_encoded']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y, random_state=0)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Best k value search
best_k = 1
best_accuracy = 0

# Trying k from 1 to 16 to find the best value of k
for k in range(1, 15):
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X_train_scaled, y_train)
    y_pred = knn_classifier.predict(X_val_scaled)
    accuracy = accuracy_score(y_val, y_pred)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k

knn_classifier_best = KNeighborsClassifier(n_neighbors=best_k)
knn_classifier_best.fit(X_train_scaled, y_train)

y_train_pred_knn = knn_classifier_best.predict(X_train_scaled)
y_val_pred_knn = knn_classifier_best.predict(X_val_scaled)

train_accuracy_knn = accuracy_score(y_train, y_train_pred_knn)
val_accuracy_knn = accuracy_score(y_val, y_val_pred_knn)

train_report = classification_report(y_train, y_train_pred_knn, output_dict=True)
val_report = classification_report(y_val, y_val_pred_knn, output_dict=True)

precision = val_report["weighted avg"]["precision"] * 100
recall = val_report["weighted avg"]["recall"] * 100
f1_score = val_report["weighted avg"]["f1-score"] * 100

# Print the results
print(f"Training Set Accuracy: {train_accuracy_knn * 100:.2f}%")
print(f"Validation Set Accuracy: {val_accuracy_knn * 100:.2f}%")
print(f"Precision: {precision:.2f}%")
print(f"Recall: {recall:.2f}%")
print(f"F1-Score: {f1_score:.2f}%")
print()

cm1 = pd.crosstab(y_train, y_train_pred_knn, rownames=['Actual'], colnames=['Predicted'], margins=True)
print('Training Confusion Matrix:\n', cm1)

print("\n")

cm2 = pd.crosstab(y_val, y_val_pred_knn, rownames=['Actual'], colnames=['Predicted'], margins=True)
print('Validation Confusion Matrix:\n', cm2)
print()  

# Calculating Testing Accuracy

test_df = pd.read_csv('./dummy_test.csv')

test_df['Pos_encoded'] = le.transform(test_df['Pos'])

test_df.drop(['Predicted Pos', 'Pos', 'Age', 'FG%', '3P%', '2P%', 'eFG%', 'FT%'], axis=1, errors='ignore', inplace=True)

X_test = test_df.drop(columns=['Pos_encoded'])
y_test = test_df['Pos_encoded']

X_test_scaled = scaler.transform(X_test)  # Using the same Scaler object as did for the Training Data
y_test_pred = knn_classifier_best.predict(X_test_scaled)

test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Testing Accuracy: {test_accuracy * 100:.3f}%")
print("\n")

print("Classification Report: \n", classification_report(y_test, y_test_pred))

# Task-3: Incorporating Cross-Validation

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

# Calculate cross-validated scores on the same knn classifier as did for training
cv_scores = cross_val_score(knn_classifier_best, X, y, cv=skf)

print(f"Cross-Validated Accuracy Scores for each fold: {cv_scores}")
print("\n")
print(f"Mean CV Accuracy of kNN : {cv_scores.mean()}")
print(f"Mean CV Accuracy of kNN Percentage Value: {cv_scores.mean() * 100:.3f}%")