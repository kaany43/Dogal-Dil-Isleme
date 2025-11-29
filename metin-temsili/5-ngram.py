from sklearn.feature_extraction.text import CountVectorizer

# Örnek metin verisi
documents = [
    "Bu bir örnek metindir",
    "Doğal dil işleme çok ilginç bir alandır",
    "Metin temsili önemlidir"
]

# N-gram (örneğin, bi-gram) temsili oluşturma
vectorizer_unigram = CountVectorizer(ngram_range=(1, 1))
vectorizer_bigram = CountVectorizer(ngram_range=(2, 2))
vectorizer_trigram = CountVectorizer(ngram_range=(3, 3))

# Metin verisini n-gram modeline dönüştürme
unigram_matrix = vectorizer_unigram.fit_transform(documents)
bigram_matrix = vectorizer_bigram.fit_transform(documents)
trigram_matrix = vectorizer_trigram.fit_transform(documents)

# N-gram matrislerini dizi formatında alma
unigram_array = unigram_matrix.toarray()
bigram_array = bigram_matrix.toarray()
trigram_array = trigram_matrix.toarray()

# Özellik isimlerini alma
unigram_features = vectorizer_unigram.get_feature_names_out()
bigram_features = vectorizer_bigram.get_feature_names_out()
trigram_features = vectorizer_trigram.get_feature_names_out()

# Sonuçları yazdırma
print("Unigram Özellik İsimleri:", unigram_features)
print("Unigram Matrisi:\n", unigram_array)
print("Bigram Özellik İsimleri:", bigram_features)
print("Bigram Matrisi:\n", bigram_array)
print("Trigram Özellik İsimleri:", trigram_features)
print("Trigram Matrisi:\n", trigram_array)
