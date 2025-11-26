import nltk

nltk.download('wordnet')

def stem_and_lemmatize_text(text):
    """
    Metindeki kelimelerin kök ve gövdelerini bulur.

    Parametreler:
    text (str): İşlenecek metin.

    Dönüş:
    tuple: Kök ve gövde listeleri.
    """
    from nltk.stem import PorterStemmer, WordNetLemmatizer

    # Stemmer ve Lemmatizer nesnelerini oluşturma
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    # Kelimeleri ayırma
    words = nltk.word_tokenize(text)

    # Kök ve gövde bulma
    stemmed_words = [stemmer.stem(word) for word in words]
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]

    return stemmed_words, lemmatized_words

# Örnek kullanım
if __name__ == "__main__":
    sample_text = "The cats are playing with the balls."
    stemmed, lemmatized = stem_and_lemmatize_text(sample_text)
    print("Kökler:", stemmed)
    print("Gövdeler:", lemmatized)
    # Çıktı:
    # Kökler: ['the', 'cat', 'are', 'play', 'with', 'the', 'ball', '.']
    # Gövdeler: ['The', 'cat', 'are', 'playing', 'with', 'the', 'ball', '.']