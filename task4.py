# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv("credit_card_fraud.csv")  # change filename if needed

# Display dataset information
print(df.info())
print(df.head())

# Check missing values
print(df.isnull().sum())

# Fill missing values if any
df.fillna(df.mean(numeric_only=True), inplace=True)

# Convert string column to numeric
le = LabelEncoder()
df['merchant_category'] = le.fit_transform(df['merchant_category'])

# Features and target
X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Apply SMOTE to handle imbalance
smote = SMOTE(random_state=42)

X_train_resampled, y_train_resampled = smote.fit_resample(
    X_train, y_train
)

print("SMOTE Applied Successfully!")
print("Before SMOTE:")
print(y_train.value_counts())

print("After SMOTE:")
print(y_train_resampled.value_counts())

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_resampled, y_train_resampled)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()