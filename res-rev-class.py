import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# 1. Load the data
dataset = pd.read_csv(r'F:\My_Projects\resturant-reviews-ml-project\scraped_df.csv')

# --- Data Preprocessing ---
# Drop rows where 'Reviews' is missing, as it's the main feature
dataset.dropna(subset=['Reviews'], inplace=True)

# Use 'hugging_face_label' for the target variable (0 or 1) and 'Reviews' as the feature
X = dataset['Reviews']
y = dataset['hugging_face_label'].astype(int)

# Check the class distribution (shows severe imbalance: 587 positive vs 92 negative)
# print(y.value_counts())

# 3. Perform Train-Test Split
# Using stratify=y to ensure the test set has the same proportion of classes as the original data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 4. Vectorize the text data (TF-IDF)
# Using TfidfVectorizer to convert text into numerical features
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

# 5. Train a Classification Model (Logistic Regression)
model = LogisticRegression(solver='liblinear', random_state=42)
model.fit(X_train_vec, y_train)

# 6. Make predictions
y_pred = model.predict(X_test_vec)

# 7. Evaluate and Visualize Results

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative (0)', 'Positive (1)'],
            yticklabels=['Negative (0)', 'Positive (1)'])
plt.title('Confusion Matrix for Restaurant Review Classification')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
# plt.savefig('confusion_matrix.png') # Saved to file by the interpreter