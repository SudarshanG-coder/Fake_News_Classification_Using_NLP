# main.py
# Fake News Classification Using NLP

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# -------------------------
# 1. Load Dataset
# -------------------------
df = pd.read_csv("data/train.csv")  # Place your CSV in a 'data' folder
df = df.fillna('')  # Handle missing values

# -------------------------
# 2. Combine Text Fields
# -------------------------
df['content'] = df['title'] + " " + df['author'] + " " + df['text']

# -------------------------
# 3. Minimal Text Preprocessing
# -------------------------
stop_words = set([
    'a', 'an', 'the', 'and', 'or', 'is', 'are', 'was', 'were', 'in', 'on', 'at',
    'to', 'for', 'of', 'by', 'with', 'about', 'as', 'from', 'up', 'down', 'out',
    'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
    'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most',
    'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
    'than', 'too', 'very', 'can', 'will', 'just', 'don', 'should', 'now'
])

def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)  # Keep only letters
    words = text.lower().split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

df['content'] = df['content'].apply(clean_text)

# -------------------------
# 4. Feature Extraction
# -------------------------
X = df['content'].values
y = df['label'].values

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# -------------------------
# 5. Train-Test Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=2
)

# -------------------------
# 6. Logistic Regression
# -------------------------
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)

# -------------------------
# 7. Passive Aggressive Classifier
# -------------------------
pa_model = PassiveAggressiveClassifier(max_iter=50)
pa_model.fit(X_train, y_train)
pa_pred = pa_model.predict(X_test)
pa_acc = accuracy_score(y_test, pa_pred)

# -------------------------
# 8. Confusion Matrix Plot
# -------------------------
cm = confusion_matrix(y_test, lr_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Real', 'Fake'],
            yticklabels=['Real', 'Fake'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Logistic Regression')
plt.tight_layout()
plt.savefig("images/confusion_matrix.png")  # Save to images folder

# -------------------------
# 9. Print Results
# -------------------------
print(f"Logistic Regression Accuracy: {lr_acc * 100:.2f}%")
print(f"Passive Aggressive Classifier Accuracy: {pa_acc * 100:.2f}%")
print("Confusion matrix saved at: images/confusion_matrix.png")
