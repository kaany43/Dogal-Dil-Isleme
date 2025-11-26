#Count Vectorizer kullanarak metin verisini Bag of Words (BoW) modeline dönüştürme
from sklearn.feature_extraction.text import CountVectorizer

#Örnek metin verisi
documents = [
    "Bu bir örnek metindir",
    "Doğal dil işleme çok ilginç bir alandır",
    "Metin temsili önemlidir"
]

#Count Vectorizer nesnesi oluşturma
vectorizer = CountVectorizer()

#Metin verisini BoW modeline dönüştürme
bow_matrix = vectorizer.fit_transform(documents)

#BoW matrisini dizi formatında alma
bow_array = bow_matrix.toarray()

#Özellik isimlerini alma
feature_names = vectorizer.get_feature_names_out()

#Sonuçları yazdırma
print("Özellik İsimleri:", feature_names)
print("Bag of Words Matrisi:\n", bow_array)
# Çıktı:
# Özellik İsimleri: ['alandır' 'bir' 'doğal' 'dil' 'işleme' 'ilginç' 'metindir' 'metin' 'önemlidir' 'temsili' 'çok']
# Bag of Words Matrisi:
#  [[0 1 0 0 0 0 1 0 0 0 0]
#  [1 1 1 1 1 1 0 0 0 0 1]
#  [0 0 0 0 0 0 0 1 1 1 0]]
