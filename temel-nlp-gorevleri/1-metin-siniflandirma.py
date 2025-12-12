'''
Spam ve Ham Sınıflandırma - Binary Text Classification with Logistic Regression
'''
import pandas as pd
from sklearn.model_selection import train_test_split    
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Veri setini yükle
data = pd.read_csv('data\\spam.csv', encoding='latin-1')
data = data[['v1', 'v2']]
data.columns = ['label', 'text']

# Label Encoding
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['label'])

# Eğitim / Test ayrımı
X_train, X_test, y_train, y_test = train_test_split(
    data['text'], data['label'], test_size=0.2, random_state=42
)

# Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

model = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1,2),
        min_df=2,
        stop_words='english'
    )),
    ('clf', LogisticRegression(
        class_weight='balanced',
        max_iter=2000
    ))
])

# Modeli eğit
model.fit(X_train, y_train)

# Test sonuçları
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Yeni metin örnekleri
new_texts = [
    "Congratulations! You've won a free ticket to Bahamas. Call now!",
    "Hey, are we still on for lunch tomorrow?"
]

preds = model.predict(new_texts)
pred_labels = label_encoder.inverse_transform(preds)

for text, label in zip(new_texts, pred_labels):
    print(f'Text: \"{text}\" => Predicted label: {label}')
