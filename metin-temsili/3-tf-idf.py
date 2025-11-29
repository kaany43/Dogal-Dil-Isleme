import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

# Ornek metin verisi
documents = [
    "Bu bir örnek metindir",
    "Doğal dil işleme çok ilginç bir alandır",
    "Metin temsili önemlidir"
]

# TF-IDF Vectorizer nesnesi oluşturma
vectorizer = TfidfVectorizer()

# Metin verisini TF-IDF modeline dönüştürme
tfidf_matrix = vectorizer.fit_transform(documents)

# TF-IDF matrisini dizi formatında alma
tfidf_array = tfidf_matrix.toarray()

# Özellik isimlerini alma
feature_names = vectorizer.get_feature_names_out()

# DataFrame oluşturma
tfidf_df = pd.DataFrame(tfidf_array, columns=feature_names)

# Sonuçları yazdırma
print("Özellik İsimleri:", feature_names)
print("TF-IDF Matrisi:\n", tfidf_df)