import nltk

from nltk.corpus import stopwords

nltk.download('stopwords')
# Kutuphane ile stop-words cikarimi
def remove_stopwords_nltk(text, language='english'):
    """
    Metinden stop-words (durak kelimeleri) çıkarır.

    Parametreler:
    text (str): İşlenecek metin.
    language (str): Stop-words dilini belirtir (varsayılan: 'english').

    Dönüş:
    str: Stop-words çıkarılmış metin.
    """
    stop_words = set(stopwords.words(language))
    words = nltk.word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

# Örnek kullanım
if __name__ == "__main__":
    sample_text = "This is a sample text with some common stop-words."
    cleaned_text = remove_stopwords_nltk(sample_text, language='english')
    print("Orijinal Metin:", sample_text)
    print("Stop-words Çıkarılmış Metin:", cleaned_text)
    # Çıktı:
    # Orijinal Metin: This is a sample text with some common stop-words.
    # Stop-words Çıkarılmış Metin: sample text common stop-words .
