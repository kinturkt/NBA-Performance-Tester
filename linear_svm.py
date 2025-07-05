import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
import sys

dataset_path = sys.argv[1]               
target_column = sys.argv[2]

df = pd.read_csv("nba_stats.csv")

le = LabelEncoder()
df['Pos_encoded'] = le.fit_transform(df['Pos'])

drop_cols = ['Age', 'FG%', '3P%', '2P%', 'eFG%', 'FT%']
X = df.drop(columns=['Pos', 'Pos_encoded'] + drop_cols, errors='ignore')
y = df['Pos_encoded']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y, shuffle=True)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

lsvm = LinearSVC (random_state=0)
lsvm.fit(X_train_scaled, y_train)

y_train_pred_lsvm = lsvm.predict(X_train_scaled)
y_val_pred_lsvm = lsvm.predict(X_val_scaled)

train_accuracy = accuracy_score(y_train, y_train_pred_lsvm)
val_accuracy = accuracy_score(y_val, y_val_pred_lsvm)

val_report = classification_report(y_val, y_val_pred_lsvm, output_dict=True)
precision = val_report["weighted avg"]["precision"] * 100
recall = val_report["weighted avg"]["recall"] * 100
f1_score = val_report["weighted avg"]["f1-score"] * 100

print(f"Training Set Accuracy: {train_accuracy * 100:.2f}%")
print(f"Validation Set Accuracy: {val_accuracy * 100:.2f}%")
print(f"Precision: {precision:.2f}%")
print(f"Recall: {recall:.2f}%")
print(f"F1-Score: {f1_score:.2f}%")

print("\nTraining Confusion Matrix:\n", pd.crosstab(y_train, y_train_pred_lsvm, rownames=['Actual'], colnames=['Predicted'], margins=True))
print("\nValidation Confusion Matrix:\n", pd.crosstab(y_val, y_val_pred_lsvm, rownames=['Actual'], colnames=['Predicted'], margins=True))

test_df = pd.read_csv("dummy_test.csv")

test_df['Pos_encoded'] = le.transform(test_df['Pos'])
test_df.drop(['Pos', 'Age', 'FG%', '3P%', '2P%', 'eFG%', 'FT%'], axis=1, errors='ignore', inplace=True)

X_test = test_df.drop(columns=['Pos_encoded'])
y_test = test_df['Pos_encoded']

X_test_scaled = scaler.transform(X_test)
y_test_pred = lsvm.predict(X_test_scaled)

test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Test Set Accuracy: {test_accuracy * 100:.2f}%")

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
cv_scores = cross_val_score(LinearSVC(random_state=0), X_train_scaled, y_train, cv=skf, scoring='accuracy')

# Print Cross-Validation Results
print("\nCross-Validation Accuracy Scores:")
for i, score in enumerate(cv_scores, 1):
    print(f"Fold {i}: {score * 100:.2f}%")

average_cv_accuracy = cv_scores.mean()
print(f"\nAverage Cross-Validation Accuracy: {average_cv_accuracy * 100:.2f}%")