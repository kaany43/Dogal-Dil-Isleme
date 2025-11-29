import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

# Veri setini yükleme
df = pd.read_csv('data/spam.csv', encoding='latin1')

# Metin temizleme 
def clean_text(text):
    text = text.lower()
    text = ''.join(char for char in text if char.isalnum() or char.isspace())
    text = ' '.join(text.split())
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Metinleri temizleme
cleaned_texts = [clean_text(text) for text in df['v2'].values]

# TF-IDF temsili oluşturma
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(cleaned_texts)
feature_names = vectorizer.get_feature_names_out()

# Kelime skorları (tüm dökümanlarda toplam TF-IDF değeri)
tfidf_scores = tfidf_matrix.sum(axis=0).A1

# Skor DataFrame'i oluşturma
df_tfidf_scores = pd.DataFrame({
    "word": feature_names,
    "tfidf_score": tfidf_scores
})

# Skora göre sıralama
df_tfidf_scores = df_tfidf_scores.sort_values(by="tfidf_score", ascending=False)

# İlk 10 kelime
print(df_tfidf_scores.head(10))
